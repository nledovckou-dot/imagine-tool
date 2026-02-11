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
_HEDRA_BASE = os.environ.get("HEDRA_BASE_URL", "https://mercury.dev.dream-ai.com/api").strip().rstrip("/")
_FAL_KEY = os.environ.get("FAL_KEY", "").strip()
_FAL_BASE = "https://queue.fal.run"
_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
_GEMINI_BASE = os.environ.get("GEMINI_BASE_URL", "").strip().rstrip("/")
_GEMINI_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")
_YANDEX_API_KEY = os.environ.get("YANDEX_GPT_API_KEY", "").strip()
_YANDEX_FOLDER = os.environ.get("YANDEX_FOLDER_ID", "").strip()
_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

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
    "— Опиши продукт с фото максимально точно: цвет, форма, текст на упаковке.\n\n"
    "ТВОЯ ЗАДАЧА — обогатить идею клиента контекстом:\n"
    "— Если упомянуто МЕСТО — добавь культуру, архитектуру, символы, природу.\n"
    "— Если упомянут ПРОДУКТ — покажи его в контексте: люди, эмоции, стиль жизни.\n"
    "— Если упомянута ТЕМА — раскрой атмосферу, субкультуру, детали.\n"
    "— Всегда добавляй ЛЮДЕЙ, АКТИВНОСТЬ, ЭМОЦИИ.\n\n"
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
            return _parse_brief(text)
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")[:300] if e.fp else ""
            last_exc = RuntimeError(f"Gemini brief HTTP {e.code}: {err}")
            if e.code == 429:
                time.sleep((2 ** attempt) * 3)  # 3, 6, 12 sec
                continue
            raise last_exc from e
        except (TimeoutError, OSError) as e:
            last_exc = RuntimeError(str(e))
            time.sleep((2 ** attempt) * 2)
            continue
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
                    return fpath
            raise RuntimeError("Gemini: no image in response")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8")[:300] if e.fp else ""
            last_exc = RuntimeError(f"Gemini HTTP {e.code}: {err}")
            if e.code in (429, 500, 502, 503):
                time.sleep((2 ** attempt) * 2)
                continue
            raise last_exc from e
        except (TimeoutError, OSError) as e:
            last_exc = RuntimeError(str(e))
            time.sleep((2 ** attempt) * 2)
            continue
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

    # v2-master supports 5 or 10 sec
    if duration_sec not in (5, 10):
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


# ── Hedra (new API at api.hedra.com) ──

_HEDRA_API = "https://api.hedra.com/web-app/public"
# Most-used video model from user's history
_HEDRA_VIDEO_MODEL = "d5af0ed3-c505-4495-802b-efb3e8ea741c"


def _hedra_request(method: str, path: str, body: Optional[dict] = None, timeout: int = 60) -> dict:
    """Make authenticated request to new Hedra API."""
    url = f"{_HEDRA_API}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-API-Key", _HEDRA_KEY)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:300] if e.fp else ""
        raise RuntimeError(f"Hedra HTTP {e.code}: {err}") from e


def _hedra_upload_image(image_path: str) -> str:
    """Upload image to Hedra and return the asset URL."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    mime = detect_mime(img_bytes)

    # Upload via multipart to assets endpoint
    boundary = f"----HedraUp{int(time.time())}"
    fname = os.path.basename(image_path)
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{fname}\"\r\nContent-Type: {mime}\r\n\r\n".encode(),
        img_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    url = f"{_HEDRA_API}/assets"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("X-API-Key", _HEDRA_KEY)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:300] if e.fp else ""
        raise RuntimeError(f"Hedra upload HTTP {e.code}: {err}") from e

    asset_id = result.get("id") or result.get("asset_id")
    if not asset_id:
        raise RuntimeError(f"Hedra upload: no asset_id: {result}")
    return asset_id


def generate_hedra_video(image_path: str, text: str) -> str:
    """Generate video via Hedra new API (image-to-video)."""
    if not _HEDRA_KEY:
        raise RuntimeError("HEDRA_API_KEY not configured")

    # Upload keyframe image
    print(f"[hedra] Uploading keyframe: {image_path}", flush=True)
    try:
        asset_id = _hedra_upload_image(image_path)
        print(f"[hedra] Keyframe asset: {asset_id}", flush=True)
    except Exception:
        asset_id = None

    # Create video generation
    payload = {
        "type": "video",
        "ai_model_id": _HEDRA_VIDEO_MODEL,
        "generated_video_inputs": {
            "text_prompt": text[:2000],
        },
        "batch_size": 1,
    }
    if asset_id:
        payload["start_keyframe_id"] = asset_id

    print(f"[hedra] Starting video generation...", flush=True)
    gen_resp = _hedra_request("POST", "/generations", body=payload, timeout=60)

    gen_id = gen_resp.get("id")
    if not gen_id:
        raise RuntimeError(f"Hedra: no generation id: {gen_resp}")

    # Poll for completion (max 5 min)
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(10)
        status = _hedra_request("GET", f"/generations/{gen_id}")
        state = status.get("status", "")

        if state == "complete":
            asset = status.get("asset", {})
            video_url = asset.get("url") or asset.get("asset", {}).get("url", "")
            if not video_url:
                # Try batch_results
                batch = status.get("batch_results", [])
                if batch:
                    video_url = batch[0].get("asset", {}).get("url", "")
            if not video_url:
                raise RuntimeError(f"Hedra: complete but no video URL: {status}")

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
            return fpath

        if state == "error":
            raise RuntimeError(f"Hedra failed: {status.get('error', status)}")

        print(f"[hedra] Status: {state}, progress: {status.get('progress', '?')}", flush=True)

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
            samples = response.get("generateVideoResponse", {}).get("generatedSamples", [])
            if not samples:
                raise RuntimeError(f"Veo3: no samples in result: {sr}")
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


# ── Background job ──

def _run_job(job_id: str, idea: str, ref_images: list[tuple[bytes, str]]):
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
        if _GEMINI_KEY:
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
        img_path = None
        img_source = ""

        # Try Gemini (FREE)
        if _GEMINI_KEY:
            try:
                emit("Генерирую картинку (Gemini)...", step="dalle")
                img_path = generate_gemini_image(img_prompt)
                img_source = "Gemini"
            except Exception as gemini_err:
                emit(f"Gemini: {str(gemini_err)[:100]}. Пробую DALL-E...", step="dalle_fallback")
        # Try DALL-E (costs $0.12)
        if not img_path and _OPENAI_KEY:
            try:
                emit("Генерирую картинку (DALL-E 3 HD)...", step="dalle")
                img_path = generate_dalle(img_prompt)
                img_source = "DALL-E"
            except Exception as dalle_err:
                emit(f"DALL-E: {str(dalle_err)[:100]}. Пробую Flux...", step="dalle_fallback")
        # Try fal.ai Flux
        if not img_path and _FAL_KEY:
            try:
                emit("Генерирую картинку (Flux Pro)...", step="dalle")
                img_path = generate_flux_image(img_prompt)
                img_source = "Flux"
            except Exception as flux_err:
                emit(f"Flux: {str(flux_err)[:100]}", step="dalle_fallback")
        if not img_path:
            raise RuntimeError("Не удалось сгенерировать картинку ни через один провайдер")

        img_name = os.path.basename(img_path)
        emit(f"Картинка готова ({img_source})!", step="dalle_done", image=f"/output/{img_name}")

        # 3. Resize for video (1080p)
        video_img = resize_for_video(img_path, 1920, 1080)

        # 4. Video — sequential: Veo 3 → Hedra → Sora → Kling
        emit("Генерирую видео (Veo 3, 1080p)...", step="video")
        video_path = None
        video_provider = ""
        video_errors = []

        # Try Veo 3 (Gemini API, free in preview)
        if _GEMINI_KEY:
            try:
                video_path = generate_veo3_video(video_img, vid_prompt)
                video_provider = "Veo 3"
            except Exception as veo_err:
                video_errors.append(f"Veo 3: {str(veo_err)[:120]}")
                emit(f"Veo 3: {str(veo_err)[:100]}. Пробую Hedra...", step="video_error")
        # Fallback: Hedra (user has credits)
        if not video_path and _HEDRA_KEY:
            try:
                emit("Генерирую видео (Hedra)...", step="video")
                video_path = generate_hedra_video(video_img, vid_prompt)
                video_provider = "Hedra"
            except Exception as hedra_err:
                video_errors.append(f"Hedra: {str(hedra_err)[:120]}")
                emit(f"Hedra: {str(hedra_err)[:100]}. Пробую Sora...", step="video_error")
        # Fallback: Sora (costs ~$1)
        if not video_path and _OPENAI_KEY:
            try:
                emit("Генерирую видео (Sora)...", step="video")
                video_path = generate_sora_video(video_img, vid_prompt, 8)
                video_provider = "Sora"
            except Exception as sora_err:
                video_errors.append(f"Sora: {str(sora_err)[:120]}")
                emit(f"Sora: {str(sora_err)[:100]}", step="video_error")
        # Fallback: Kling
        if not video_path and _KLING_ACCESS and _KLING_SECRET:
            try:
                emit("Генерирую видео (Kling)...", step="video")
                video_path = generate_kling_video(video_img, vid_prompt, 10)
                video_provider = "Kling"
            except Exception as kling_err:
                video_errors.append(f"Kling: {str(kling_err)[:120]}")
                emit(f"Kling: {str(kling_err)[:100]}", step="video_error")

        if not video_path:
            raise RuntimeError("Ни одно видео не удалось:\n" + "\n".join(video_errors))

        vid_name = os.path.basename(video_path)
        videos = [{"provider": video_provider, "url": f"/output/{vid_name}"}]

        emit(
            f"Готово! Видео: {video_provider}",
            step="done",
            videos=videos,
            video=videos[0]["url"],
            video_provider=video_provider,
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
            "provider": video_provider,
            "ts": int(time.time() * 1000),
        })

    except Exception as exc:
        emit(f"Ошибка: {exc}", step="error")
        job["status"] = "error"
    finally:
        q.put(None)  # signal end


# ── Routes ──

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(_OUTPUT_DIR, filename)


@app.route("/api/generate", methods=["POST"])
def api_generate():
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

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "status": "running",
        "queue": queue.Queue(),
    }

    t = threading.Thread(target=_run_job, args=(job_id, idea, ref_images), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def api_stream(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    def generate():
        q: queue.Queue = job["queue"]
        while True:
            try:
                event = q.get(timeout=120)
            except queue.Empty:
                yield "data: {\"msg\": \"timeout\"}\n\n"
                break
            if event is None:
                break
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/history")
def api_history():
    return jsonify(_load_history())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
