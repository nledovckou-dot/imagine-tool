"""Imagine Tool — web app: text (+ ref photos) → GPT → DALL-E → Sora/Kling video."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import queue
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import List, Optional, Tuple

from flask import Flask, Response, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")

# ── Config ──

_OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
_KLING_ACCESS = os.environ.get("KLING_ACCESS_KEY", "").strip()
_KLING_SECRET = os.environ.get("KLING_SECRET_KEY", "").strip()
_HEDRA_KEY = os.environ.get("HEDRA_API_KEY", "").strip()
_FAL_KEY = os.environ.get("FAL_KEY", "").strip()
_FAL_BASE = "https://queue.fal.run"
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
_GEMINI_BASE = os.environ.get("GEMINI_BASE_URL", "").strip().rstrip("/")
_GEMINI_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
_KREA_KEY = os.environ.get("KREA_API_KEY", "").strip()
_APP_PUBLIC_URL = (os.environ.get("APP_PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")
_YANDEX_API_KEY = os.environ.get("YANDEX_GPT_API_KEY", "").strip()
_YANDEX_FOLDER = os.environ.get("YANDEX_FOLDER_ID", "").strip()
_APP_PASSWORD = os.environ.get("APP_PASSWORD", "123321")
_TG_ALERT_TOKEN = os.environ.get("IMAGINE_BOT_TOKEN", "").strip()
_TG_ALERT_CHAT_IDS = [
    c.strip() for c in os.environ.get("TG_ADMIN_USER_ID", "").split(",") if c.strip()
]
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ── Brute-force protection ──

_MAX_ATTEMPTS = 5
_LOCKOUT_SEC = 60
_login_attempts: dict[str, list[float]] = {}   # IP → failed timestamps
_login_lockout: dict[str, float] = {}           # IP → locked until
_login_lock = threading.Lock()


def _check_login(ip: str) -> tuple[bool, str]:
    """Check if IP is allowed to attempt login."""
    with _login_lock:
        now = time.time()
        if ip in _login_lockout and now < _login_lockout[ip]:
            remaining = int(_login_lockout[ip] - now)
            return False, f"Слишком много попыток. Подожди {remaining} сек."
        # Clear expired lockout
        if ip in _login_lockout and now >= _login_lockout[ip]:
            _login_lockout.pop(ip, None)
            _login_attempts.pop(ip, None)
        return True, ""


def _record_fail(ip: str) -> int:
    """Record failed attempt. Returns attempts left. Triggers lockout + TG alert at limit."""
    with _login_lock:
        now = time.time()
        if ip not in _login_attempts:
            _login_attempts[ip] = []
        _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < 300]
        _login_attempts[ip].append(now)
        count = len(_login_attempts[ip])
        if count >= _MAX_ATTEMPTS:
            _login_lockout[ip] = now + _LOCKOUT_SEC
            _login_attempts[ip] = []
            threading.Thread(
                target=_send_tg_alert,
                args=(f"⚠️ Imagine Tool: {_MAX_ATTEMPTS} неудачных попыток входа с IP {ip}. "
                      f"Заблокирован на {_LOCKOUT_SEC} сек.",),
                daemon=True,
            ).start()
            return 0
        return _MAX_ATTEMPTS - count


def _record_success(ip: str):
    """Clear attempts on successful login."""
    with _login_lock:
        _login_attempts.pop(ip, None)
        _login_lockout.pop(ip, None)


def _send_tg_alert(text: str):
    """Send alert to all admin TG chats."""
    if not _TG_ALERT_TOKEN or not _TG_ALERT_CHAT_IDS:
        print(f"[alert] No TG config: {text}", flush=True)
        return
    for chat_id in _TG_ALERT_CHAT_IDS:
        try:
            payload = json.dumps({"chat_id": chat_id, "text": text}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{_TG_ALERT_TOKEN}/sendMessage",
                data=payload, method="POST",
            )
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
            print(f"[alert] Sent to {chat_id}", flush=True)
        except Exception as e:
            print(f"[alert] TG send failed ({chat_id}): {e}", flush=True)

# ── Rate limiter for Gemini free tier ──

class RateLimiter:
    """Thread-safe rate limiter: max N requests per window_sec."""
    def __init__(self, rpm: int = 4, rpd: int = 90):
        self._rpm = rpm
        self._rpd = rpd
        self._lock = threading.Lock()
        self._minute_ts: list[float] = []
        self._day_ts: list[float] = []

    def wait_if_needed(self) -> bool:
        """Wait if rate limit would be exceeded. Returns False if daily limit hit."""
        with self._lock:
            now = time.time()
            # Clean old entries
            self._minute_ts = [t for t in self._minute_ts if now - t < 60]
            self._day_ts = [t for t in self._day_ts if now - t < 86400]
            # Check daily limit
            if len(self._day_ts) >= self._rpd:
                return False
            # Wait for minute window if needed
            if len(self._minute_ts) >= self._rpm:
                wait = 60 - (now - self._minute_ts[0]) + 0.5
                if wait > 0:
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    now = time.time()
                    self._minute_ts = [t for t in self._minute_ts if now - t < 60]
            self._minute_ts.append(now)
            self._day_ts.append(now)
            return True

_gemini_limiter = RateLimiter(rpm=4, rpd=90)  # conservative: 4 RPM, 90 RPD

# ── Circuit breaker ──

class CircuitBreaker:
    """Disable provider after consecutive failures, auto-recover after cooldown."""
    def __init__(self, max_failures: int = 3, cooldown_sec: int = 300):
        self._max = max_failures
        self._cooldown = cooldown_sec
        self._lock = threading.Lock()
        self._failures: dict[str, int] = {}
        self._disabled_until: dict[str, float] = {}

    def is_open(self, provider: str) -> bool:
        with self._lock:
            until = self._disabled_until.get(provider, 0)
            if until and time.time() < until:
                return True
            if until and time.time() >= until:
                # Cooldown passed, reset
                self._failures[provider] = 0
                self._disabled_until[provider] = 0
            return False

    def record_success(self, provider: str):
        with self._lock:
            self._failures[provider] = 0
            self._disabled_until[provider] = 0

    def record_failure(self, provider: str):
        with self._lock:
            self._failures[provider] = self._failures.get(provider, 0) + 1
            if self._failures[provider] >= self._max:
                self._disabled_until[provider] = time.time() + self._cooldown
                print(f"[circuit] {provider} disabled for {self._cooldown}s after {self._max} failures", flush=True)

_breaker = CircuitBreaker(max_failures=3, cooldown_sec=300)

# ── Provider requirements (pre-validation) ──

_PROVIDER_REQS = {
    "sora":  {"w": 1280, "h": 720,  "max_prompt": 4000},
    "kling": {"w": 1920, "h": 1080, "max_prompt": 2500},
    "hedra": {"w": 1920, "h": 1080, "max_prompt": 2000},
    "veo3":  {"w": 1920, "h": 1080, "max_prompt": 5000},
    "krea":  {"w": 1280, "h": 720,  "max_prompt": 4000},
}

_RU_VIDEO_SUFFIX = (
    "\n\nКРИТИЧЕСКИ ВАЖНО: ВСЕ надписи, титры, слоганы, субтитры, текст на экране — "
    "СТРОГО НА РУССКОМ ЯЗЫКЕ (кириллица). Речь и голос — на русском языке. "
    "ЗАПРЕЩЕНО использовать английский язык, латиницу или любые другие языки. "
    "Язык видео: РУССКИЙ. "
    "Люди в видео — соответствуют контексту рекламы (если детский лагерь — дети, "
    "если фитнес — спортивные взрослые и т.д.). Внешность людей — славянская/европейская "
    "(реклама для России). "
    "Никаких выдуманных логотипов — только реальные бренды с исходного фото."
)


def _moderation_prescreen(prompt: str) -> str:
    """Check prompt via Gemini Flash and auto-fix moderation issues. Returns cleaned prompt."""
    if not _GEMINI_KEY:
        return prompt
    base_url = _GEMINI_BASE or "https://generativelanguage.googleapis.com"
    url = f"{base_url}/v1beta/models/gemini-2.0-flash:generateContent"
    system_msg = (
        "Ты — модератор контента для AI-генерации картинок и видео.\n"
        "Проверь промпт и исправь ТОЛЬКО реально опасные элементы:\n\n"
        "ИСПРАВЛЯТЬ (заблокирует модерация):\n"
        "- насилие, кровь, оружие → динамичное действие без крови\n"
        "- нагота, сексуальный контент → убрать\n"
        "- известные реальные люди → безымянные стильные персонажи\n"
        "- медицинские процедуры крупно → общий план\n\n"
        "НЕ ТРОГАТЬ (это безопасно и пройдёт модерацию):\n"
        "- Дети в безопасном контексте (лагерь, школа, спорт, семья, игры) — "
        "это НОРМАЛЬНО, НЕ заменяй на взрослых!\n"
        "- Еда, напитки, продукты\n"
        "- Спорт, активный отдых\n\n"
        "ВАЖНО: Сохраняй культурный контекст. Если промпт на русском — "
        "люди должны выглядеть как жители России (славянская внешность).\n\n"
        "Если промпт безопасен — верни его БЕЗ ИЗМЕНЕНИЙ.\n"
        "Отвечай ТОЛЬКО текстом промпта, без пояснений."
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_msg}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 2048},
    }
    headers = {"x-goog-api-key": _GEMINI_KEY, "Content-Type": "application/json"}
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        cleaned = "".join(p.get("text", "") for p in parts).strip()
        if cleaned and len(cleaned) > 50:
            if cleaned != prompt:
                print(f"[moderation] Prompt auto-fixed ({len(prompt)}→{len(cleaned)} chars)", flush=True)
            return cleaned
    except Exception as e:
        print(f"[moderation] Prescreen failed: {e}, using original prompt", flush=True)
    return prompt


def _check_provider_balance(provider: str) -> tuple[bool, str]:
    """Check if provider has enough credits. Returns (available, reason)."""
    try:
        if provider == "kling":
            token = _kling_jwt()
            req = urllib.request.Request("https://api.klingai.com/v1/account/credits", method="GET")
            req.add_header("Authorization", f"Bearer {token}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            balance = data.get("data", {}).get("balance", 0)
            if balance <= 0:
                return False, f"Kling баланс: {balance}"
            return True, ""
        elif provider == "hedra":
            credits_resp = _hedra_request("GET", "/billing/credits")
            remaining = credits_resp.get("remaining", 0)
            if remaining < 24:  # MiniMax costs 24 (cheapest)
                return False, f"Hedra кредиты: {remaining} (нужно минимум 24)"
            return True, ""
        elif provider == "krea":
            # Krea doesn't have a balance check endpoint, just try
            return True, ""
    except Exception:
        pass
    return True, ""  # If check fails, try anyway


def _try_with_retry(fn, args, name: str, emit_fn, max_retries: int = 1) -> str:
    """Try video provider with retry on transient errors. Returns file path or raises."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args)
        except RuntimeError as e:
            last_err = e
            err_lower = str(e).lower()
            # Non-retryable — stop immediately
            if any(x in err_lower for x in [
                "not configured", "balance", "insufficient", "credits", "quota",
            ]):
                raise
            if attempt < max_retries:
                wait = 5 * (attempt + 1)
                emit_fn(f"{name}: ошибка ({str(e)[:60]}), повтор #{attempt+2} через {wait}с...", step="video_retry")
                time.sleep(wait)
    raise last_err


# ── Job tracking ──

_jobs: dict[str, dict] = {}  # job_id → {status, progress_queue, result, ...}
_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "output", "history.json")
_history_lock = threading.Lock()


def _load_history() -> list:
    try:
        with open(_HISTORY_FILE, "r") as f:
            return json.loads(f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_history_entry(entry: dict):
    with _history_lock:
        history = _load_history()
        history.insert(0, entry)
        if len(history) > 100:
            history = history[:100]
        fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(history, f, ensure_ascii=False)
            os.replace(tmp, _HISTORY_FILE)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def _openai_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_OPENAI_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "ImagineTool/1.0",
    }


# ── Gemini creative prompts (FREE tier) ──

_BRIEF_SYSTEM = (
    "Ты — креативный директор премиального рекламного агентства и культуролог. "
    "Клиент даёт короткую идею (и возможно фото-референсы бренда/продукта), "
    "а ты создаёшь ДВА промпта для генерации рекламных материалов.\n\n"
    "ВАЖНЕЙШЕЕ ПРАВИЛО — ЯЗЫК:\n"
    "— Все промпты пиши СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
    "— Любой текст, слоган, надпись в картинке/видео — ТОЛЬКО НА РУССКОМ.\n"
    "— Исключение: только если клиент ЯВНО попросил другой язык в своём промпте.\n\n"
    "ВАЖНЕЙШЕЕ ПРАВИЛО — ВЕРНОСТЬ БРЕНДУ:\n"
    "— Если клиент прикрепил фото-референсы — это РЕАЛЬНЫЙ ПРОДУКТ/БРЕНД.\n"
    "— Сохраняй ТОЧНЫЕ цвета бренда, форму продукта, логотип, упаковку.\n"
    "— Не выдумывай новый дизайн — используй то, что на фото.\n"
    "— Это реклама для конечного потребителя с ГОТОВЫМ продуктом.\n"
    "— Опиши продукт с фото максимально точно: цвет, форма, текст на упаковке.\n"
    "— НИКОГДА не выдумывай логотипы, бренды, торговые марки, которых нет на фото.\n"
    "— Если на фото НЕТ логотипа — НЕ добавляй никакой логотип.\n"
    "— Если на фото ЕСТЬ логотип — используй ТОЛЬКО его, точно как на фото.\n\n"
    "ВАЖНЕЙШЕЕ ПРАВИЛО — КОНТЕКСТ И КУЛЬТУРА:\n"
    "— СТРОГО следуй контексту запроса клиента. Если реклама про детский лагерь — "
    "показывай детей. Если про фитнес — показывай спортивных людей. НЕ заменяй "
    "целевую аудиторию на другую.\n"
    "— Если текст на русском языке — это реклама для России. "
    "Люди должны выглядеть как жители России: славянская/европейская внешность, "
    "светлая кожа, русые/тёмные волосы. НЕ генерируй случайную этничность.\n"
    "— Обстановка, архитектура, природа — российская (если не указано иное).\n"
    "— Если клиент упоминает конкретный город/регион — используй его культуру.\n\n"
    "ПРАВИЛА МОДЕРАЦИИ (для прохождения модерации AI-систем):\n"
    "— ЗАПРЕЩЕНО: насилие, оружие, кровь, наготу, сексуальный контент.\n"
    "— ЗАПРЕЩЕНО: известные реальные люди (политики, звёзды) без разрешения.\n"
    "— ЗАПРЕЩЕНО: медицинские процедуры крупным планом, шприцы, операции.\n"
    "— ЗАПРЕЩЕНО: религиозные символы в коммерческом/неуважительном контексте.\n"
    "— Дети в рекламе РАЗРЕШЕНЫ, если контекст безопасный (образование, спорт, "
    "отдых, семья). Показывай детей в АКТИВНОСТИ: бегают, учатся, играют. "
    "Избегай крупных планов лиц детей — лучше общие/средние планы групп.\n\n"
    "ТВОЯ ЗАДАЧА — обогатить идею клиента контекстом:\n"
    "— Если упомянуто МЕСТО — добавь культуру, архитектуру, символы, природу.\n"
    "— Если упомянут ПРОДУКТ — покажи его в контексте: люди, эмоции, стиль жизни.\n"
    "— Если упомянута ТЕМА — раскрой атмосферу, субкультуру, детали.\n"
    "— Добавляй ЛЮДЕЙ, АКТИВНОСТЬ, ЭМОЦИИ — соответствующих контексту запроса.\n\n"
    "ПРОМПТ 1 — КАРТИНКА:\n"
    "  Ключевой рекламный кадр. Премиальная эстетика.\n"
    "  Композиция, цвета, свет, текстуры, настроение.\n"
    "  Если есть референсы — ТОЧНО опиши продукт с фото (цвета, форма, лого).\n"
    "  Любой текст/слоган — НА РУССКОМ.\n"
    "  800–1200 символов.\n\n"
    "ПРОМПТ 2 — ВИДЕО (8 сек кинематографичный рекламный ролик):\n"
    "  Описывай посекундно как режиссёр:\n"
    "  — 0-2 сек: экспозиция, общий план, движение камеры\n"
    "  — 2-4 сек: средний план, действие, динамика\n"
    "  — 4-6 сек: крупный план, кульминация, эмоции\n"
    "  — 6-8 сек: финал, продукт крупно, слоган НА РУССКОМ\n"
    "  Если есть референсы — продукт должен выглядеть ТОЧНО как на фото.\n"
    "  Движение камеры, свет, атмосфера, цветовая палитра.\n"
    "  ОБЯЗАТЕЛЬНО ВКЛЮЧИ В ПРОМПТ ФРАЗУ: «Все надписи, титры и текст на экране "
    "написаны НА РУССКОМ ЯЗЫКЕ кириллицей. Язык видео: русский.»\n"
    "  Это КРИТИЧЕСКИ важно — видео-модели по умолчанию генерируют английский текст, "
    "поэтому русский язык нужно указать ЯВНО в промпте.\n"
    "  1000–2000 символов.\n\n"
    "ФОРМАТ ОТВЕТА (строго):\n"
    "КАРТИНКА: [промпт на русском]\n"
    "ВИДЕО: [промпт на русском]"
)


def _parse_brief(text: str) -> tuple[str, str]:
    """Parse КАРТИНКА/ВИДЕО from LLM response."""
    img_prompt = text
    vid_prompt = text
    if "КАРТИНКА:" in text and "ВИДЕО:" in text:
        parts = text.split("ВИДЕО:")
        img_prompt = parts[0].replace("КАРТИНКА:", "").strip()
        vid_prompt = parts[1].strip()
    return img_prompt, vid_prompt


def gemini_creative_prompts(idea: str, ref_images: list[tuple[bytes, str]] | None = None) -> tuple[str, str]:
    """Generate creative brief via Gemini Flash (FREE tier)."""
    if not _GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    base_url = _GEMINI_BASE or "https://generativelanguage.googleapis.com"

    # Build parts: system instruction + ref images + idea
    parts = []
    if ref_images:
        for img_bytes, mime in ref_images[:3]:
            b64 = base64.b64encode(img_bytes).decode()
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    parts.append({"text": f"Идея клиента: {idea}"})

    payload = {
        "system_instruction": {"parts": [{"text": _BRIEF_SYSTEM}]},
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 3000},
    }
    data = json.dumps(payload).encode("utf-8")
    url = f"{base_url}/v1beta/models/gemini-2.0-flash:generateContent"

    # Rate limit check
    if not _gemini_limiter.wait_if_needed():
        raise RuntimeError("Gemini: дневной лимит запросов исчерпан")

    last_exc = None
    for attempt in range(3):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("x-goog-api-key", _GEMINI_KEY)
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            candidates = body.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini brief: no candidates")
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if not text:
                raise RuntimeError("Gemini brief: empty response")
            _breaker.record_success("gemini_brief")
            return _parse_brief(text)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")[:300] if e.fp else ""
            last_exc = RuntimeError(f"Gemini brief HTTP {e.code}: {err}")
            if e.code == 429:
                _gemini_limiter.wait_if_needed()  # extra wait
                time.sleep((2 ** attempt) * 3)
                continue
            _breaker.record_failure("gemini_brief")
            raise last_exc from e
        except (TimeoutError, OSError) as e:
            last_exc = RuntimeError(str(e))
            _breaker.record_failure("gemini_brief")
            time.sleep((2 ** attempt) * 2)
            continue
    _breaker.record_failure("gemini_brief")
    raise last_exc or RuntimeError("Gemini brief: all retries failed")


# ── GPT creative prompts (fallback) ──

def gpt_creative_prompts(idea: str, ref_images: list[tuple[bytes, str]] | None = None) -> tuple[str, str]:
    """Generate creative brief via OpenAI GPT (fallback, costs money)."""
    user_parts: list = []
    if ref_images:
        for img_bytes, mime in ref_images[:3]:
            b64 = base64.b64encode(img_bytes).decode()
            user_parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"}})
        user_parts.append({"type": "text", "text": f"Идея клиента: {idea}"})
        user_content = user_parts
    else:
        user_content = f"Идея клиента: {idea}"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": _BRIEF_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 2500,
        "temperature": 0.7,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{_OPENAI_BASE}/chat/completions", data=data, method="POST")
    for k, v in _openai_headers().items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:500] if e.fp else ""
        raise RuntimeError(f"GPT HTTP {e.code}: {err[:200]}") from e
    text = body["choices"][0]["message"]["content"].strip()
    return _parse_brief(text)


def yandex_gpt_creative_prompts(idea: str) -> tuple[str, str]:
    """Fallback: use YandexGPT for creative brief if OpenAI is down."""
    if not _YANDEX_API_KEY or not _YANDEX_FOLDER:
        raise RuntimeError("YandexGPT not configured")

    payload = {
        "modelUri": f"gpt://{_YANDEX_FOLDER}/yandexgpt/latest",
        "completionOptions": {"maxTokens": 2500, "temperature": 0.7},
        "messages": [
            {"role": "system", "text": (
                "Ты — креативный директор. Клиент даёт идею, ты создаёшь 2 промпта.\n"
                "ПРОМПТ 1 — КАРТИНКА: детальное описание ключевого кадра, 1000-1500 символов. "
                "Композиция, свет, цвета, люди, эмоции, текстуры.\n"
                "ПРОМПТ 2 — ВИДЕО: посекундное описание 8-секундного ролика, 1500-2500 символов. "
                "Движение камеры, смена планов, действие, финал.\n"
                "ФОРМАТ:\nКАРТИНКА: [промпт]\nВИДЕО: [промпт]"
            )},
            {"role": "user", "text": f"Идея: {idea}"},
        ],
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        data=body, method="POST",
    )
    req.add_header("Authorization", f"Api-Key {_YANDEX_API_KEY}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:300] if e.fp else ""
        raise RuntimeError(f"YandexGPT HTTP {e.code}: {err}") from e

    text = result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "")
    if not text:
        raise RuntimeError(f"YandexGPT: empty response")

    img_prompt = text
    vid_prompt = text
    if "КАРТИНКА:" in text and "ВИДЕО:" in text:
        parts = text.split("ВИДЕО:")
        img_prompt = parts[0].replace("КАРТИНКА:", "").strip()
        vid_prompt = parts[1].strip()
    return img_prompt, vid_prompt


def fallback_prompts(idea: str) -> tuple[str, str]:
    """Last resort: use raw idea with template enhancement."""
    img_prompt = (
        f"Премиальная рекламная фотография высочайшего качества. {idea}. "
        "Кинематографичная композиция, профессиональный студийный свет, "
        "глубина резкости, яркие насыщенные цвета, люди в кадре с эмоциями, "
        "детализированные текстуры, 8k, hyperrealistic, editorial photography, "
        "стиль Apple/Nike/Mercedes рекламы."
    )
    vid_prompt = (
        f"8-секундный кинематографичный рекламный ролик премиум-класса. {idea}. "
        "Секунды 0-2: общий план, плавный наезд камеры, устанавливаем сцену. "
        "Секунды 2-4: средний план, герои в действии, динамика нарастает. "
        "Секунды 4-6: крупные планы, детали, кульминация движения, slow motion. "
        "Секунды 6-8: финальный кадр, логотип/слоган появляется. "
        "Движение камеры: crane shot, dolly zoom, golden hour свет, "
        "кинематографичная цветокоррекция, shallow depth of field, "
        "частицы в воздухе, блики, драматические тени."
    )
    return img_prompt, vid_prompt


# ── Gemini image generation (works from Russia via proxy) ──

def generate_gemini_image(prompt: str) -> str:
    """Generate image via Google Gemini (needs GEMINI_BASE_URL proxy from Russia)."""
    if not _GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")
    if not _GEMINI_BASE:
        raise RuntimeError("GEMINI_BASE_URL not configured")

    url = f"{_GEMINI_BASE}/v1beta/models/{_GEMINI_MODEL}:generateContent"
    headers = {
        "x-goog-api-key": _GEMINI_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    data = json.dumps(payload).encode("utf-8")

    # Rate limit check
    if not _gemini_limiter.wait_if_needed():
        raise RuntimeError("Gemini image: дневной лимит запросов исчерпан")

    last_exc = None
    for attempt in range(3):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            candidates = body.get("candidates") or []
            if not candidates:
                raise RuntimeError("Gemini: no candidates")
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    img_bytes = base64.b64decode(inline["data"])
                    mime = inline.get("mimeType", "image/png")
                    ext = ".jpg" if "jpeg" in mime else ".png"
                    fpath = os.path.join(_OUTPUT_DIR, f"gemini_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}")
                    fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=ext)
                    try:
                        with os.fdopen(fd, "wb") as f:
                            f.write(img_bytes)
                        os.replace(tmp, fpath)
                    except BaseException:
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
                        raise
                    _breaker.record_success("gemini_image")
                    return fpath
            raise RuntimeError("Gemini: no image in response")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")[:300] if e.fp else ""
            last_exc = RuntimeError(f"Gemini HTTP {e.code}: {err}")
            if e.code in (429, 500, 502, 503):
                _gemini_limiter.wait_if_needed()
                time.sleep((2 ** attempt) * 2)
                continue
            _breaker.record_failure("gemini_image")
            raise last_exc from e
        except (TimeoutError, OSError) as e:
            last_exc = RuntimeError(str(e))
            _breaker.record_failure("gemini_image")
            time.sleep((2 ** attempt) * 2)
            continue
    _breaker.record_failure("gemini_image")
    raise last_exc or RuntimeError("Gemini: all retries failed")


# ── fal.ai Flux (image fallback) ──

def generate_flux_image(prompt: str) -> str:
    """Generate image via fal.ai Flux Pro."""
    if not _FAL_KEY:
        raise RuntimeError("FAL_KEY not configured")

    payload = {
        "prompt": prompt[:4000],
        "image_size": "landscape_16_9",
        "num_images": 1,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{_FAL_BASE}/fal-ai/flux-pro/v1.1",
        data=body, method="POST",
    )
    req.add_header("Authorization", f"Key {_FAL_KEY}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:500] if e.fp else ""
        raise RuntimeError(f"Flux submit HTTP {e.code}: {err[:200]}") from e

    request_id = result.get("request_id")
    if not request_id:
        # Synchronous response — check for images directly
        images = result.get("images", [])
        if images:
            return _download_fal_image(images[0].get("url", ""))
        raise RuntimeError(f"Flux: no request_id or images: {result}")

    print(f"[flux] Submitted job: {request_id}", flush=True)

    # Poll for completion (max 2 min)
    status_url = f"{_FAL_BASE}/fal-ai/flux-pro/v1.1/requests/{request_id}/status"
    result_url = f"{_FAL_BASE}/fal-ai/flux-pro/v1.1/requests/{request_id}"

    deadline = time.time() + 120
    while time.time() < deadline:
        time.sleep(5)
        poll_req = urllib.request.Request(status_url, method="GET")
        poll_req.add_header("Authorization", f"Key {_FAL_KEY}")
        try:
            with urllib.request.urlopen(poll_req, timeout=30) as resp:
                sr = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            continue
        status = sr.get("status", "")
        if status == "COMPLETED":
            break
        if status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"Flux failed: {sr}")
    else:
        raise RuntimeError("Flux: таймаут (2 мин)")

    res_req = urllib.request.Request(result_url, method="GET")
    res_req.add_header("Authorization", f"Key {_FAL_KEY}")
    with urllib.request.urlopen(res_req, timeout=30) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    images = res.get("images", [])
    if not images:
        raise RuntimeError(f"Flux: no images in result: {res}")
    return _download_fal_image(images[0].get("url", ""))


def _download_fal_image(img_url: str) -> str:
    """Download image from fal.ai URL."""
    if not img_url:
        raise RuntimeError("Flux: empty image URL")
    with urllib.request.urlopen(img_url, timeout=60) as resp:
        img_bytes = resp.read()
    ext = ".png" if "png" in img_url else ".jpg"
    fpath = os.path.join(_OUTPUT_DIR, f"flux_{int(time.time())}_{uuid.uuid4().hex[:6]}{ext}")
    fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=ext)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(img_bytes)
        os.replace(tmp, fpath)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return fpath


# ── DALL-E ──

def generate_dalle(prompt: str) -> str:
    prompt = prompt[:4000]
    payload = {"model": "dall-e-3", "prompt": prompt, "n": 1, "size": "1792x1024", "quality": "hd"}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{_OPENAI_BASE}/images/generations", data=data, method="POST")
    for k, v in _openai_headers().items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:500] if e.fp else ""
        raise RuntimeError(f"DALL-E HTTP {e.code}: {err[:200]}") from e
    img_url = body["data"][0]["url"]
    with urllib.request.urlopen(img_url, timeout=60) as resp:
        img_bytes = resp.read()
    fpath = os.path.join(_OUTPUT_DIR, f"img_{int(time.time())}_{uuid.uuid4().hex[:6]}.png")
    fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(img_bytes)
        os.replace(tmp, fpath)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return fpath


# ── Resize ──

def resize_for_video(image_path: str, width: int = 1280, height: int = 720) -> str:
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    out_path = os.path.splitext(image_path)[0] + f"_{width}x{height}.jpg"
    fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=".jpg")
    try:
        with os.fdopen(fd, "wb") as f:
            img.save(f, format="JPEG", quality=90)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return out_path


# ── Sora 2 ──

def detect_mime(data: bytes) -> str:
    if data[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return "image/webp"
    return "image/jpeg"


def generate_sora_video(image_path: str, prompt: str, seconds: int = 8) -> str:
    with open(image_path, "rb") as f:
        img_data = f.read()
    mime = detect_mime(img_data)

    boundary = f"----SoraBound{int(time.time())}"
    fields = {
        "model": "sora-2",
        "prompt": prompt,
        "size": "1280x720",
        "seconds": str(seconds),
    }
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    ext_map = {"image/jpeg": "ref.jpg", "image/png": "ref.png", "image/webp": "ref.webp"}
    fname = ext_map.get(mime, "ref.png")
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"input_reference\"; "
        f"filename=\"{fname}\"\r\nContent-Type: {mime}\r\n\r\n".encode()
    )
    parts.append(img_data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    url = f"{_OPENAI_BASE}/videos"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {_OPENAI_KEY}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("User-Agent", "ImagineTool/1.0")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err = e.read().decode()[:500] if e.fp else ""
        raise RuntimeError(f"Sora API HTTP {e.code}: {err[:200]}") from e

    video_id = result.get("id")
    if not video_id:
        raise RuntimeError(f"Sora: no video ID: {result}")

    deadline = time.time() + 120  # 2 min max — fallback to Kling if slow
    while time.time() < deadline:
        time.sleep(10)
        poll_req = urllib.request.Request(f"{_OPENAI_BASE}/videos/{video_id}", method="GET")
        poll_req.add_header("Authorization", f"Bearer {_OPENAI_KEY}")
        poll_req.add_header("User-Agent", "ImagineTool/1.0")
        try:
            with urllib.request.urlopen(poll_req, timeout=30) as resp:
                pr = json.loads(resp.read().decode())
        except urllib.error.HTTPError:
            continue

        st = pr.get("status", "")
        if st == "completed":
            break
        if st == "failed":
            err_msg = pr.get("error", {}).get("message", str(pr))
            raise RuntimeError(f"Sora failed: {err_msg}")
    else:
        raise RuntimeError("Sora: слишком долго, переключаюсь на Kling")

    dl_req = urllib.request.Request(f"{_OPENAI_BASE}/videos/{video_id}/content", method="GET")
    dl_req.add_header("Authorization", f"Bearer {_OPENAI_KEY}")
    dl_req.add_header("User-Agent", "ImagineTool/1.0")

    with urllib.request.urlopen(dl_req, timeout=300) as resp:
        video_bytes = resp.read()

    if len(video_bytes) < 1000:
        raise RuntimeError(f"Sora: file too small ({len(video_bytes)} bytes)")

    fpath = os.path.join(_OUTPUT_DIR, f"sora_{video_id}.mp4")
    fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(video_bytes)
        os.replace(tmp, fpath)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return fpath


# ── Kling (fallback) ──

def _kling_jwt() -> str:
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    now = int(time.time())
    payload_data = {"iss": _KLING_ACCESS, "exp": now + 1800, "nbf": now - 5, "iat": now}
    payload = base64.urlsafe_b64encode(
        json.dumps(payload_data).encode()
    ).rstrip(b"=").decode()
    signing_input = f"{header}.{payload}".encode()
    signature = base64.urlsafe_b64encode(
        hmac.new(_KLING_SECRET.encode(), signing_input, hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return f"{header}.{payload}.{signature}"


def generate_kling_video(image_path: str, prompt: str, duration_sec: int = 10) -> str:
    if not _KLING_ACCESS or not _KLING_SECRET:
        raise RuntimeError("Kling API keys not configured")

    # Read raw file bytes and encode to base64 (no PIL re-compression)
    with open(image_path, "rb") as f:
        raw_bytes = f.read()
    image_b64 = base64.b64encode(raw_bytes).decode()
    print(f"[kling] Image: {image_path}, size={len(raw_bytes)} bytes, b64={len(image_b64)} chars", flush=True)

    # Kling v2: 5 or 10 sec
    if duration_sec < 5:
        duration_sec = 5
    if duration_sec > 10:
        duration_sec = 10
    # Kling API accepts raw base64 or URL (NOT data URI with prefix)
    body_data = {
        "model_name": "kling-v2-master",
        "image": image_b64,
        "prompt": prompt,
        "duration": str(duration_sec),
        "mode": "std",
        "cfg_scale": 0.5,
    }

    token = _kling_jwt()
    data = json.dumps(body_data).encode("utf-8")
    req = urllib.request.Request(
        "https://api.klingai.com/v1/videos/image2video",
        data=data, method="POST",
    )
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:500] if e.fp else ""
        raise RuntimeError(f"Kling HTTP {e.code}: {err[:200]}") from e

    if result.get("code") != 0:
        raise RuntimeError(f"Kling error: {result.get('message', result)}")

    task_id = result.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"Kling: no task_id: {result}")

    deadline = time.time() + 180  # 3 min for Kling
    while time.time() < deadline:
        time.sleep(8)
        token = _kling_jwt()
        poll_req = urllib.request.Request(
            f"https://api.klingai.com/v1/videos/image2video/{task_id}",
            method="GET",
        )
        poll_req.add_header("Authorization", f"Bearer {token}")
        poll_req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(poll_req, timeout=30) as resp:
            sr = json.loads(resp.read().decode("utf-8"))
        sd = sr.get("data", {})
        ts = sd.get("task_status", "")
        if ts == "succeed":
            videos = sd.get("task_result", {}).get("videos", [])
            if not videos:
                raise RuntimeError("Kling: no videos in result")
            video_url = videos[0].get("url", "")
            if not video_url:
                raise RuntimeError("Kling: empty video URL")
            with urllib.request.urlopen(video_url, timeout=120) as vr:
                vbytes = vr.read()
            fpath = os.path.join(_OUTPUT_DIR, f"kling_{task_id}.mp4")
            fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=".mp4")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(vbytes)
                os.replace(tmp, fpath)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            return fpath
        if ts == "failed":
            raise RuntimeError(f"Kling failed: {sd.get('task_status_msg', 'unknown')}")
    raise RuntimeError("Kling: таймаут (3 мин)")


# ── Hedra (new API at api.hedra.com — official starter format) ──

_HEDRA_API = "https://api.hedra.com/web-app/public"
_HEDRA_VIDEO_MODEL = os.environ.get("HEDRA_VIDEO_MODEL", "fb657777-6b02-478d-87a9-e02e8c53748c")  # Veo 3 I2V

# Per-model valid durations (from /models API)
_HEDRA_DURATIONS = {
    "fb657777": 8000,   # Veo 3: 4000/6000/8000
    "9963e814": 8000,   # Veo 3 Fast: 4000/6000/8000
    "b917e7da": 6000,   # MiniMax Hailuo: 6000/10000
    "0e451fde": 5000,   # Kling 2.5 Turbo: 5000/10000
}


def _hedra_request(method: str, path: str, body: Optional[dict] = None, timeout: int = 60) -> dict:
    """Make authenticated JSON request to Hedra API."""
    url = f"{_HEDRA_API}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-API-Key", _HEDRA_KEY)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:500] if e.fp else ""
        raise RuntimeError(f"Hedra HTTP {e.code}: {err}") from e


def _hedra_upload_image(image_path: str) -> str:
    """Upload image to Hedra via 2-step flow: create asset → upload file."""
    fname = os.path.basename(image_path)

    # Step 1: Create asset record
    create_resp = _hedra_request("POST", "/assets", body={"name": fname, "type": "image"})
    asset_id = create_resp.get("id")
    if not asset_id:
        raise RuntimeError(f"Hedra: no asset id from create: {create_resp}")
    print(f"[hedra] Created asset: {asset_id}", flush=True)

    # Step 2: Upload file to the asset
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    boundary = f"----HedraUp{int(time.time())}"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{fname}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode(),
        img_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    url = f"{_HEDRA_API}/assets/{asset_id}/upload"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("X-API-Key", _HEDRA_KEY)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()  # consume response
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:300] if e.fp else ""
        raise RuntimeError(f"Hedra upload HTTP {e.code}: {err}") from e

    print(f"[hedra] Uploaded image to asset {asset_id}", flush=True)
    return asset_id


def generate_hedra_video(image_path: str, text: str) -> str:
    """Generate video via Hedra API (image-to-video)."""
    if not _HEDRA_KEY:
        raise RuntimeError("HEDRA_API_KEY not configured")

    # Upload keyframe image (2-step)
    print(f"[hedra] Uploading keyframe: {image_path}", flush=True)
    image_id = _hedra_upload_image(image_path)

    # Create video generation — auto-select model based on available credits
    model_id = os.environ.get("_HEDRA_MODEL_OVERRIDE", _HEDRA_VIDEO_MODEL)

    # If no override, check credits and pick affordable model
    if not os.environ.get("_HEDRA_MODEL_OVERRIDE"):
        try:
            credits_resp = _hedra_request("GET", "/billing/credits")
            remaining = credits_resp.get("remaining", 9999)
            # Model costs: Veo3=440, Veo3Fast=160, Kling=50, MiniMax=24
            _MODEL_COSTS = [
                ("fb657777", 440),  # Veo 3
                ("9963e814", 160),  # Veo 3 Fast
                ("0e451fde", 50),   # Kling 2.5 Turbo
                ("b917e7da", 24),   # MiniMax Hailuo
            ]
            for mid_prefix, cost in _MODEL_COSTS:
                if model_id.startswith(mid_prefix) and remaining < cost:
                    # Current model too expensive, downgrade
                    for alt_prefix, alt_cost in _MODEL_COSTS:
                        if remaining >= alt_cost:
                            full_id = next(
                                (f"{p}-{_HEDRA_VIDEO_MODEL.split('-', 1)[1]}" for p in [alt_prefix]),
                                None,
                            )
                            # Find full ID from HEDRA_DURATIONS keys
                            for dur_prefix in _HEDRA_DURATIONS:
                                if dur_prefix.startswith(alt_prefix):
                                    break
                            # Use the cheapest affordable model from known IDs
                            _CHEAP_MODELS = {
                                "b917e7da": "b917e7da-f0a4-42d1-b52f-67ee11569cc8",
                                "0e451fde": "0e451fde-9e6f-48e6-83a9-222f6cc05eba",
                                "9963e814": "9963e814-d1ee-4518-a844-7ed380ddbb20",
                                "fb657777": "fb657777-6b02-478d-87a9-e02e8c53748c",
                            }
                            model_id = _CHEAP_MODELS.get(alt_prefix, model_id)
                            print(f"[hedra] Credits={remaining}, auto-downgraded to {alt_prefix} (cost={alt_cost})", flush=True)
                            break
                    break
        except Exception:
            pass  # If credits check fails, proceed with default

    # Pick valid duration for this model
    duration = 8000
    for prefix, dur in _HEDRA_DURATIONS.items():
        if model_id.startswith(prefix):
            duration = dur
            break
    payload = {
        "type": "video",
        "ai_model_id": model_id,
        "start_keyframe_id": image_id,
        "generated_video_inputs": {
            "text_prompt": text[:2000],
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "duration_ms": duration,
        },
        "batch_size": 1,
    }

    print(f"[hedra] Starting video generation (model={model_id})...", flush=True)
    gen_resp = _hedra_request("POST", "/generations", body=payload, timeout=60)

    gen_id = gen_resp.get("id")
    if not gen_id:
        raise RuntimeError(f"Hedra: no generation id: {gen_resp}")
    print(f"[hedra] Generation ID: {gen_id}", flush=True)

    # Poll for completion via /status endpoint (max 5 min)
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(8)
        status = _hedra_request("GET", f"/generations/{gen_id}/status")
        state = status.get("status", "")

        if state == "complete":
            video_url = status.get("download_url") or status.get("url", "")
            if not video_url:
                raise RuntimeError(f"Hedra: complete but no download_url: {status}")

            with urllib.request.urlopen(video_url, timeout=120) as resp:
                vbytes = resp.read()
            fpath = os.path.join(_OUTPUT_DIR, f"hedra_{gen_id[:12]}.mp4")
            fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=".mp4")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(vbytes)
                os.replace(tmp, fpath)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            print(f"[hedra] Downloaded video: {fpath} ({len(vbytes)} bytes)", flush=True)
            return fpath

        if state == "error":
            err_msg = status.get("error_message") or status.get("error", str(status))
            raise RuntimeError(f"Hedra failed: {err_msg}")

        print(f"[hedra] Status: {state}", flush=True)

    raise RuntimeError("Hedra: таймаут (5 мин)")


# ── Veo 3.1 via Google Gemini API (direct, no FAL.ai) ──

def generate_veo3_video(image_path: str, prompt: str) -> str:
    """Generate video via Google Veo 3.1 through Gemini API (image-to-video)."""
    if not _GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured")

    # Read image as base64
    with open(image_path, "rb") as f:
        raw_bytes = f.read()
    image_b64 = base64.b64encode(raw_bytes).decode()
    mime = detect_mime(raw_bytes)
    print(f"[veo3] Image: {len(raw_bytes)} bytes, mime={mime}", flush=True)

    # Gemini API base (direct or proxy)
    base_url = _GEMINI_BASE or "https://generativelanguage.googleapis.com"

    # Submit long-running video generation
    payload = {
        "instances": [{
            "prompt": prompt,
            "image": {"bytesBase64Encoded": image_b64, "mimeType": mime},
        }],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "16:9",
            "durationSeconds": 8,
            "resolution": "1080p",
        },
    }
    body = json.dumps(payload).encode("utf-8")
    url = f"{base_url}/v1beta/models/veo-3.1-generate-preview:predictLongRunning"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("x-goog-api-key", _GEMINI_KEY)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:500] if e.fp else ""
        raise RuntimeError(f"Veo3 submit HTTP {e.code}: {err[:200]}") from e

    operation_name = result.get("name")
    if not operation_name:
        raise RuntimeError(f"Veo3: no operation name: {result}")
    print(f"[veo3] Operation: {operation_name}", flush=True)

    # Poll for completion (max 5 min)
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(10)
        poll_url = f"{base_url}/v1beta/{operation_name}"
        poll_req = urllib.request.Request(poll_url, method="GET")
        poll_req.add_header("x-goog-api-key", _GEMINI_KEY)
        try:
            with urllib.request.urlopen(poll_req, timeout=30) as resp:
                sr = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            continue

        if sr.get("done"):
            # Extract video URI
            response = sr.get("response", {})
            gen_resp = response.get("generateVideoResponse", {})
            # Check for content moderation filter
            rai_reasons = gen_resp.get("raiMediaFilteredReasons", [])
            if rai_reasons:
                raise RuntimeError(f"Veo3 модерация: {'; '.join(rai_reasons)}")
            if gen_resp.get("raiMediaFilteredCount"):
                raise RuntimeError("Veo3: контент заблокирован модерацией (дети, насилие и т.д.)")
            samples = gen_resp.get("generatedSamples", [])
            if not samples:
                raise RuntimeError(f"Veo3: нет результата: {sr}")
            video_uri = samples[0].get("video", {}).get("uri", "")
            if not video_uri:
                raise RuntimeError(f"Veo3: no video URI: {samples[0]}")

            # Download video (needs API key for Google-hosted URIs)
            dl_req = urllib.request.Request(video_uri, method="GET")
            dl_req.add_header("x-goog-api-key", _GEMINI_KEY)
            with urllib.request.urlopen(dl_req, timeout=120) as resp:
                vbytes = resp.read()

            op_short = operation_name.split("/")[-1][:12]
            fpath = os.path.join(_OUTPUT_DIR, f"veo3_{op_short}.mp4")
            fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=".mp4")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(vbytes)
                os.replace(tmp, fpath)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            return fpath

        if sr.get("error"):
            raise RuntimeError(f"Veo3 failed: {sr['error'].get('message', sr['error'])}")

        print(f"[veo3] Polling... done={sr.get('done', False)}", flush=True)

    raise RuntimeError("Veo3: таймаут (5 мин)")


# ── Krea AI (aggregator: Veo, Kling, Hailuo, Wan — all via one API) ──

def generate_krea_video(image_path: str, prompt: str, duration: int = 6) -> str:
    """Generate video via Krea AI image-to-video API (MiniMax Hailuo)."""
    if not _KREA_KEY:
        raise RuntimeError("KREA_API_KEY not configured")
    if not _APP_PUBLIC_URL:
        raise RuntimeError("APP_PUBLIC_URL not configured (Krea needs public image URL)")

    img_name = os.path.basename(image_path)
    image_url = f"{_APP_PUBLIC_URL}/output/{img_name}"

    # MiniMax Hailuo via Krea: only 6 or 10 allowed (API enum)
    if duration >= 8:
        duration = 10
    else:
        duration = 6

    payload = {
        "startImage": image_url,
        "prompt": prompt[:4000],
        "duration": duration,
    }
    data = json.dumps(payload).encode("utf-8")
    url = "https://api.krea.ai/generate/video/minimax/hailuo-2.3"
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {_KREA_KEY}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
    req.add_header("Accept", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:500] if e.fp else ""
        raise RuntimeError(f"Krea API HTTP {e.code}: {err[:200]}") from e

    # Krea returns generation object with id
    job_id = result.get("id") or result.get("job_id")
    if not job_id:
        raise RuntimeError(f"Krea: no job id: {json.dumps(result)[:300]}")
    print(f"[krea] Job: {job_id}", flush=True)

    # Poll for completion (max 5 min)
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(8)
        poll_url = f"https://api.krea.ai/jobs/{job_id}"
        poll_req = urllib.request.Request(poll_url, method="GET")
        poll_req.add_header("Authorization", f"Bearer {_KREA_KEY}")
        poll_req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        try:
            with urllib.request.urlopen(poll_req, timeout=30) as resp:
                sr = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            continue

        status = sr.get("status", "")
        if status in ("completed", "succeeded") or (sr.get("completed_at") and status != "failed"):
            # Krea returns result.urls[] array
            result_obj = sr.get("result") or {}
            urls = result_obj.get("urls") or []
            video_url = urls[0] if urls else ""
            # Fallback: try other shapes
            if not video_url:
                video_url = (
                    result_obj.get("video_url", "")
                    or sr.get("video_url", "")
                    or sr.get("url", "")
                )
            if not video_url:
                raise RuntimeError(f"Krea: no video_url in response: {json.dumps(sr)[:400]}")
            with urllib.request.urlopen(video_url, timeout=120) as vr:
                vbytes = vr.read()
            fpath = os.path.join(_OUTPUT_DIR, f"krea_{str(job_id)[:12]}.mp4")
            fd, tmp = tempfile.mkstemp(dir=_OUTPUT_DIR, suffix=".mp4")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(vbytes)
                os.replace(tmp, fpath)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            print(f"[krea] Downloaded: {fpath} ({len(vbytes)} bytes)", flush=True)
            return fpath

        if status == "failed":
            err_msg = sr.get("error", sr.get("message", str(sr)))
            raise RuntimeError(f"Krea failed: {err_msg}")

        print(f"[krea] Polling... status={status}", flush=True)

    raise RuntimeError("Krea: таймаут (5 мин)")


# ── Background job ──

def _run_job(job_id: str, idea: str, ref_images: list[tuple[bytes, str]], video_provider: str = "veo3"):
    job = _jobs[job_id]
    q: queue.Queue = job["queue"]

    def emit(msg: str, **extra):
        event = {"msg": msg, **extra}
        q.put(event)

    try:
        # 1. Creative brief (Gemini FREE → GPT → YandexGPT → template)
        img_prompt = vid_prompt = ""
        brief_source = ""
        emit("Создаю креативный бриф...", step="gpt")

        # Try Gemini Flash (FREE)
        if _GEMINI_KEY and not _breaker.is_open("gemini_brief"):
            try:
                img_prompt, vid_prompt = gemini_creative_prompts(idea, ref_images if ref_images else None)
                brief_source = "Gemini"
            except Exception as gem_err:
                emit(f"Gemini бриф: {str(gem_err)[:100]}. Пробую GPT...", step="gpt_fallback")
        # Try OpenAI GPT (costs money)
        if not img_prompt and _OPENAI_KEY:
            try:
                img_prompt, vid_prompt = gpt_creative_prompts(idea, ref_images if ref_images else None)
                brief_source = "GPT"
            except Exception as gpt_err:
                emit(f"GPT: {str(gpt_err)[:100]}. Пробую YandexGPT...", step="gpt_fallback")
        # Try YandexGPT
        if not img_prompt and _YANDEX_API_KEY:
            try:
                img_prompt, vid_prompt = yandex_gpt_creative_prompts(idea)
                brief_source = "YandexGPT"
            except Exception as ya_err:
                emit(f"YandexGPT: {str(ya_err)[:100]}. Использую шаблон...", step="gpt_fallback")
        # Fallback to template
        if not img_prompt:
            img_prompt, vid_prompt = fallback_prompts(idea)
            brief_source = "шаблон"

        emit(f"Бриф готов ({brief_source})", step="gpt_done", img_prompt=img_prompt, vid_prompt=vid_prompt)

        # 2. Image (Gemini FREE → DALL-E → Flux)
        # Add moderation suffix to image prompt to prevent content policy violations
        _IMG_MODERATION = (
            "\n\nВАЖНО: Люди на картинке соответствуют контексту рекламы "
            "(если реклама для детей — показывай детей, если для взрослых — взрослых). "
            "Внешность людей — славянская/европейская (реклама для российского рынка). "
            "Никаких выдуманных логотипов или брендов — только то, что есть в описании."
        )
        img_prompt_safe = img_prompt + _IMG_MODERATION

        img_path = None
        img_source = ""

        # Try Gemini (FREE)
        if _GEMINI_KEY and not _breaker.is_open("gemini_image"):
            try:
                emit("Генерирую картинку (Gemini)...", step="dalle")
                img_path = generate_gemini_image(img_prompt_safe)
                img_source = "Gemini"
            except Exception as gemini_err:
                emit(f"Gemini: {str(gemini_err)[:100]}. Пробую DALL-E...", step="dalle_fallback")
        # Try DALL-E (costs $0.12)
        if not img_path and _OPENAI_KEY:
            try:
                emit("Генерирую картинку (DALL-E 3 HD)...", step="dalle")
                img_path = generate_dalle(img_prompt_safe)
                img_source = "DALL-E"
            except Exception as dalle_err:
                emit(f"DALL-E: {str(dalle_err)[:100]}. Пробую Flux...", step="dalle_fallback")
        # Try fal.ai Flux
        if not img_path and _FAL_KEY:
            try:
                emit("Генерирую картинку (Flux Pro)...", step="dalle")
                img_path = generate_flux_image(img_prompt_safe)
                img_source = "Flux"
            except Exception as flux_err:
                emit(f"Flux: {str(flux_err)[:100]}", step="dalle_fallback")
        if not img_path:
            raise RuntimeError("Не удалось сгенерировать картинку ни через один провайдер")

        img_name = os.path.basename(img_path)
        emit(f"Картинка готова ({img_source})!", step="dalle_done", image=f"/output/{img_name}")

        # 3. Moderation prescreen + Russian language + per-provider image resize
        emit("Проверяю промпт на модерацию...", step="moderation")
        vid_prompt_clean = _moderation_prescreen(vid_prompt)
        vid_prompt_ru = vid_prompt_clean + _RU_VIDEO_SUFFIX

        # Pre-validate: resize per provider requirements (cache unique sizes)
        _resized_cache: dict[tuple[int,int], str] = {}
        def _get_resized(w: int, h: int) -> str:
            key = (w, h)
            if key not in _resized_cache:
                _resized_cache[key] = resize_for_video(img_path, w, h)
            return _resized_cache[key]

        # Build provider args with correct sizes and prompt limits
        def _build_args(pkey: str) -> tuple:
            reqs = _PROVIDER_REQS.get(pkey, {"w": 1920, "h": 1080, "max_prompt": 4000})
            img = _get_resized(reqs["w"], reqs["h"])
            prompt = vid_prompt_ru[:reqs["max_prompt"]]
            if pkey == "sora":
                return (img, prompt, 8)
            elif pkey == "kling":
                return (img, prompt, 10)
            elif pkey == "krea":
                return (img, prompt, 8)
            return (img, prompt)

        # 4. Video — with auto-fallback chain + retry
        _HEDRA_MODELS = {
            "hedra": ("fb657777-6b02-478d-87a9-e02e8c53748c", "Hedra Veo3"),
            "hedra_minimax": ("b917e7da-f0a4-42d1-b52f-67ee11569cc8", "Hedra MiniMax"),
            "hedra_kling": ("0e451fde-9e6f-48e6-83a9-222f6cc05eba", "Hedra Kling"),
            "hedra_veo3fast": ("9963e814-d1ee-4518-a844-7ed380ddbb20", "Hedra Veo3 Fast"),
        }

        def _hedra_with_model(model_id: str, img: str, prompt: str) -> str:
            old_model = os.environ.get("_HEDRA_MODEL_OVERRIDE")
            os.environ["_HEDRA_MODEL_OVERRIDE"] = model_id
            try:
                return generate_hedra_video(img, prompt)
            finally:
                if old_model:
                    os.environ["_HEDRA_MODEL_OVERRIDE"] = old_model
                else:
                    os.environ.pop("_HEDRA_MODEL_OVERRIDE", None)

        # Provider map: key → (display_name, function)
        _provider_fns = {
            "veo3": ("Veo 3", generate_veo3_video),
            "hedra": ("Hedra", generate_hedra_video),
            "sora": ("Sora", generate_sora_video),
            "kling": ("Kling", generate_kling_video),
            "krea": ("Krea", generate_krea_video),
        }
        _provider_map = {}
        for _pk, (_pn, _pfn) in _provider_fns.items():
            _provider_map[_pk] = (_pn, _pfn, _build_args(_pk))
        _provider_available = {
            "veo3": bool(_GEMINI_KEY),
            "hedra": bool(_HEDRA_KEY),
            "sora": bool(_OPENAI_KEY),
            "kling": bool(_KLING_ACCESS and _KLING_SECRET),
            "krea": bool(_KREA_KEY and _APP_PUBLIC_URL),
        }

        # Pre-flight balance checks — skip providers with 0 credits
        for _bpk in list(_provider_available.keys()):
            if _provider_available[_bpk]:
                ok, reason = _check_provider_balance(_bpk)
                if not ok:
                    _provider_available[_bpk] = False
                    emit(f"{_bpk}: пропущен ({reason})", step="balance_check")
                    print(f"[balance] {_bpk} skipped: {reason}", flush=True)

        # Fallback chains: if primary fails, try these next
        _fallback_chain = {
            "veo3": ["hedra", "hedra_veo3fast", "krea", "sora"],
            "hedra": ["hedra_minimax", "hedra_kling", "hedra_veo3fast", "krea", "veo3"],
            "sora": ["hedra", "hedra_veo3fast", "krea", "veo3"],
            "kling": ["hedra_kling", "hedra", "hedra_minimax", "krea"],
            "krea": ["hedra", "hedra_minimax", "veo3", "sora"],
        }

        if video_provider == "all":
            # Run all available providers in parallel (no fallback for parallel)
            chosen = [k for k in _provider_map if _provider_available.get(k)]
        else:
            chosen = [video_provider] if _provider_available.get(video_provider) else []

        if not chosen:
            raise RuntimeError(f"Провайдер '{video_provider}' не настроен")

        video_errors = []

        if len(chosen) == 1:
            # Single provider with auto-fallback
            pkey = chosen[0]
            pname, pfn, pargs = _provider_map[pkey]
            videos = []

            # Try primary with retry
            if not _breaker.is_open(pkey):
                emit(f"Генерирую видео ({pname})...", step="video")
                try:
                    vpath = _try_with_retry(pfn, pargs, pname, emit, max_retries=1)
                    vid_name = os.path.basename(vpath)
                    videos = [{"provider": pname, "url": f"/output/{vid_name}"}]
                    _breaker.record_success(pkey)
                except Exception as e:
                    _breaker.record_failure(pkey)
                    video_errors.append(f"{pname}: {str(e)[:120]}")
                    emit(f"{pname} не удалось: {str(e)[:80]}. Пробую запасной...", step="video_error")
            else:
                emit(f"{pname} временно отключён (3 ошибки подряд). Пробую запасной...", step="video_error")

            # Auto-fallback chain
            if not videos and _HEDRA_KEY:
                fallbacks = _fallback_chain.get(pkey, [])
                for fb_key in fallbacks:
                    if _breaker.is_open(fb_key):
                        continue
                    if fb_key in _HEDRA_MODELS:
                        model_id, fb_name = _HEDRA_MODELS[fb_key]
                        hedra_args = _build_args("hedra")
                        emit(f"Пробую {fb_name}...", step="video")
                        try:
                            vpath = _try_with_retry(
                                _hedra_with_model, (model_id, hedra_args[0], hedra_args[1]),
                                fb_name, emit, max_retries=1,
                            )
                            vid_name = os.path.basename(vpath)
                            videos = [{"provider": fb_name, "url": f"/output/{vid_name}"}]
                            _breaker.record_success(fb_key)
                            break
                        except Exception as e:
                            _breaker.record_failure(fb_key)
                            video_errors.append(f"{fb_name}: {str(e)[:120]}")
                            emit(f"{fb_name}: {str(e)[:80]}", step="video_error")
                    elif fb_key in _provider_map and _provider_available.get(fb_key):
                        fb_name, fb_fn, fb_args = _provider_map[fb_key]
                        emit(f"Пробую {fb_name}...", step="video")
                        try:
                            vpath = _try_with_retry(fb_fn, fb_args, fb_name, emit, max_retries=1)
                            vid_name = os.path.basename(vpath)
                            videos = [{"provider": fb_name, "url": f"/output/{vid_name}"}]
                            _breaker.record_success(fb_key)
                            break
                        except Exception as e:
                            _breaker.record_failure(fb_key)
                            video_errors.append(f"{fb_name}: {str(e)[:120]}")
                            emit(f"{fb_name}: {str(e)[:80]}", step="video_error")
        else:
            # Multiple providers — parallel (no fallback)
            names = [_provider_map[k][0] for k in chosen]
            emit(f"Генерирую видео ({' + '.join(names)} параллельно)...", step="video")
            results: dict[str, dict] = {}
            lock = threading.Lock()

            def run_prov(name, fn, args):
                try:
                    path = _try_with_retry(fn, args, name, emit, max_retries=1)
                    with lock:
                        results[name] = {"path": path}
                    emit(f"{name}: готово!", step="video_partial")
                except Exception as e:
                    with lock:
                        results[name] = {"error": str(e)}
                    emit(f"{name}: {str(e)[:120]}", step="video_error")

            threads = []
            for pkey in chosen:
                pname, pfn, pargs = _provider_map[pkey]
                t = threading.Thread(target=run_prov, args=(pname, pfn, pargs))
                threads.append(t)
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=360)

            videos = []
            for name, res in results.items():
                if "path" in res:
                    vid_name = os.path.basename(res["path"])
                    videos.append({"provider": name, "url": f"/output/{vid_name}"})
                else:
                    video_errors.append(f"{name}: {res.get('error', 'unknown')}")

        if not videos:
            raise RuntimeError("Ни одно видео не удалось:\n" + "\n".join(video_errors))

        result_provider = ", ".join(v["provider"] for v in videos)

        emit(
            f"Готово! Видео: {result_provider}",
            step="done",
            videos=videos,
            video=videos[0]["url"],
            video_provider=result_provider,
            image=f"/output/{img_name}",
            vid_prompt=vid_prompt,
            errors=video_errors if video_errors else None,
        )
        job["status"] = "done"

        # Save to history
        _save_history_entry({
            "idea": idea,
            "img_prompt": img_prompt,
            "vid_prompt": vid_prompt,
            "image": f"/output/{img_name}",
            "video": videos[0]["url"],
            "videos": videos,
            "provider": result_provider,
            "ts": int(time.time() * 1000),
        })

    except Exception as exc:
        emit(f"Ошибка: {exc}", step="error")
        job["status"] = "error"
    finally:
        q.put(None)  # signal end


# ── Video-only regen job ──

def _run_regen_job(job_id: str, img_path: str, vid_prompt: str, provider: str):
    """Re-generate video only (skip brief + image)."""
    job = _jobs[job_id]
    q: queue.Queue = job["queue"]

    def emit(msg: str, **extra):
        q.put({"msg": msg, **extra})

    try:
        vid_prompt_clean = _moderation_prescreen(vid_prompt)
        vid_prompt_ru = vid_prompt_clean if _RU_VIDEO_SUFFIX in vid_prompt_clean else vid_prompt_clean + _RU_VIDEO_SUFFIX

        _provider_fns = {
            "veo3": ("Veo 3", generate_veo3_video),
            "hedra": ("Hedra", generate_hedra_video),
            "sora": ("Sora", generate_sora_video),
            "kling": ("Kling", generate_kling_video),
            "krea": ("Krea", generate_krea_video),
        }

        if provider not in _provider_fns:
            raise RuntimeError(f"Неизвестный провайдер: {provider}")

        pname, pfn = _provider_fns[provider]
        reqs = _PROVIDER_REQS.get(provider, {"w": 1920, "h": 1080, "max_prompt": 4000})
        resized = resize_for_video(img_path, reqs["w"], reqs["h"])
        prompt = vid_prompt_ru[:reqs["max_prompt"]]

        if provider == "sora":
            args = (resized, prompt, 8)
        elif provider == "kling":
            args = (resized, prompt, 10)
        elif provider == "krea":
            args = (resized, prompt, 8)
        else:
            args = (resized, prompt)

        emit(f"Генерирую видео ({pname})...", step="video")
        vpath = _try_with_retry(pfn, args, pname, emit, max_retries=1)
        vid_name = os.path.basename(vpath)
        img_name = os.path.basename(img_path)

        videos = [{"provider": pname, "url": f"/output/{vid_name}"}]
        emit(
            f"Готово! Видео: {pname}",
            step="done",
            videos=videos,
            video=videos[0]["url"],
            video_provider=pname,
            image=f"/output/{img_name}",
            vid_prompt=vid_prompt,
        )
        job["status"] = "done"

    except Exception as exc:
        emit(f"Ошибка: {exc}", step="error")
        job["status"] = "error"
    finally:
        q.put(None)


# ── Routes ──

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(_OUTPUT_DIR, filename)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    ip = ip.split(",")[0].strip()  # first IP if behind proxy

    # Brute-force check
    allowed, reason = _check_login(ip)
    if not allowed:
        return jsonify({"error": reason}), 429

    # Password check
    password = request.form.get("password", "").strip()
    if password != _APP_PASSWORD:
        left = _record_fail(ip)
        if left <= 0:
            return jsonify({"error": f"Неверный пароль. Заблокировано на {_LOCKOUT_SEC} сек."}), 429
        return jsonify({"error": f"Неверный пароль. Осталось попыток: {left}"}), 403

    _record_success(ip)

    if not _OPENAI_KEY and not _GEMINI_KEY and not _FAL_KEY:
        return jsonify({"error": "No API keys configured (need GEMINI_API_KEY, OPENAI_API_KEY or FAL_KEY)"}), 500

    idea = request.form.get("idea", "").strip()
    if not idea:
        return jsonify({"error": "idea is required"}), 400

    # Collect uploaded reference photos
    ref_images = []
    for key in sorted(request.files.keys()):
        if key.startswith("photo"):
            f = request.files[key]
            data = f.read()
            if data:
                mime = detect_mime(data)
                ref_images.append((data, mime))

    video_provider = request.form.get("video_provider", "veo3").strip()

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "queue": queue.Queue(),
    }

    t = threading.Thread(target=_run_job, args=(job_id, idea, ref_images, video_provider), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/regen-video", methods=["POST"])
def api_regen_video():
    """Re-generate video only with a different provider (reuses existing image)."""
    ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    ip = ip.split(",")[0].strip()

    allowed, reason = _check_login(ip)
    if not allowed:
        return jsonify({"error": reason}), 429

    password = request.form.get("password", "").strip()
    if password != _APP_PASSWORD:
        left = _record_fail(ip)
        if left <= 0:
            return jsonify({"error": f"Заблокировано на {_LOCKOUT_SEC} сек."}), 429
        return jsonify({"error": f"Неверный пароль. Осталось: {left}"}), 403

    _record_success(ip)

    image = request.form.get("image", "").strip()
    vid_prompt = request.form.get("vid_prompt", "").strip()
    provider = request.form.get("provider", "").strip()

    if not image or not vid_prompt or not provider:
        return jsonify({"error": "image, vid_prompt, provider обязательны"}), 400

    img_name = os.path.basename(image)
    img_path = os.path.join(_OUTPUT_DIR, img_name)
    if not os.path.isfile(img_path):
        return jsonify({"error": "Картинка не найдена"}), 404

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "queue": queue.Queue()}

    t = threading.Thread(target=_run_regen_job, args=(job_id, img_path, vid_prompt, provider), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    def generate():
        q: queue.Queue = job["queue"]
        start = time.time()
        while time.time() - start < 600:  # 10 min max total
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                # Send SSE comment as keepalive (ignored by EventSource, keeps connection alive)
                yield ": keepalive\n\n"
                continue
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/history")
def api_history():
    return jsonify(_load_history())


@app.route("/api/health")
def api_health():
    """Health check: provider status, credits, circuit breaker state."""
    providers = {}

    # Gemini
    providers["gemini_brief"] = {
        "configured": bool(_GEMINI_KEY),
        "circuit_open": _breaker.is_open("gemini_brief"),
        "rate_limit": f"{len(_gemini_limiter._minute_ts)}/{_gemini_limiter._rpm} RPM, "
                      f"{len(_gemini_limiter._day_ts)}/{_gemini_limiter._rpd} RPD",
    }
    providers["gemini_image"] = {
        "configured": bool(_GEMINI_KEY),
        "circuit_open": _breaker.is_open("gemini_image"),
    }

    # Hedra — check credits
    hedra_credits = None
    if _HEDRA_KEY:
        try:
            resp_data = _hedra_request("GET", "/billing/credits")
            hedra_credits = resp_data
        except Exception:
            hedra_credits = "error checking"
    providers["hedra"] = {
        "configured": bool(_HEDRA_KEY),
        "circuit_open": _breaker.is_open("hedra"),
        "credits": hedra_credits,
    }

    # OpenAI (Sora + DALL-E)
    providers["sora"] = {
        "configured": bool(_OPENAI_KEY),
        "circuit_open": _breaker.is_open("sora"),
    }
    providers["dalle"] = {
        "configured": bool(_OPENAI_KEY),
    }

    # Kling
    providers["kling"] = {
        "configured": bool(_KLING_ACCESS and _KLING_SECRET),
        "circuit_open": _breaker.is_open("kling"),
    }

    return jsonify({
        "status": "ok",
        "providers": providers,
        "active_jobs": sum(1 for j in _jobs.values() if j.get("status") == "running"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
