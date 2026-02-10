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


# ── GPT creative prompts ──

def gpt_creative_prompts(idea: str, ref_images: list[tuple[bytes, str]] | None = None) -> tuple[str, str]:
    system_msg = (
        "Ты — креативный директор премиального рекламного агентства и культуролог. "
        "Клиент даёт короткую идею, а ты ГЛУБОКО раскрываешь тему и создаёшь ДВА промпта.\n\n"
        "ТВОЯ ГЛАВНАЯ ЗАДАЧА — обогатить идею клиента контекстом:\n"
        "— Если упомянуто МЕСТО (город, страна, регион) — добавь культуру, архитектуру, "
        "  национальные символы, природу, традиции, историю этого места.\n"
        "— Если упомянут ПРОДУКТ — покажи его не просто как вещь, а в контексте: "
        "  люди, эмоции, стиль жизни, история бренда, среда использования.\n"
        "— Если упомянута ТЕМА (спорт, музыка, еда) — раскрой атмосферу, субкультуру, "
        "  легенды, героев, характерные детали.\n"
        "— Всегда добавляй ЛЮДЕЙ, АКТИВНОСТЬ, ЭМОЦИИ — не просто пейзаж или предмет.\n\n"
        "Примеры обогащения:\n"
        "  «горы Осетии» → горные аулы, сторожевые башни, осетинские пироги на столе, "
        "  старейшина с кинжалом, туман над ущельем, всадники, нартский эпос\n"
        "  «кофейня» → бариста с латте-артом, утренний свет через витраж, пар над чашкой, "
        "  люди за ноутбуками, аромат жареных зёрен, винтажная кофемолка\n"
        "  «фитнес» → капли пота в слоу-мо, мощные мышцы, утренняя пробежка по набережной, "
        "  дыхание на морозе, момент победы, зеркала зала\n\n"
        "ПРОМПТ 1 — КАРТИНКА (для DALL-E 3):\n"
        "  Ключевой кадр. Премиальная эстетика (Mercedes, Apple, Nike). "
        "  Композиция, цвета, свет, текстуры, настроение. "
        "  Обязательно с людьми/активностью, не просто пейзаж. "
        "  Если уместно — текст/слоган. На РУССКОМ. 1000–1500 символов — "
        "  чем подробнее описание, тем лучше качество.\n\n"
        "ПРОМПТ 2 — ВИДЕО (для Sora, 8 сек кинематографичный рекламный ролик):\n"
        "  ВАЖНО: это САМЫЙ ГЛАВНЫЙ промпт, от него зависит качество видео!\n"
        "  Пиши МАКСИМАЛЬНО ПОДРОБНО, 1500–2500 символов.\n"
        "  Описывай КАЖДУЮ СЕКУНДУ ролика как режиссёр:\n"
        "  — Секунды 0-2: что видим, какой план, движение камеры\n"
        "  — Секунды 2-4: смена плана, новое действие\n"
        "  — Секунды 4-6: кульминация, максимум динамики\n"
        "  — Секунды 6-8: финал, слоган, эмоция\n"
        "  Обязательно включи:\n"
        "  — Движение камеры (наезд, облёт, панорама, слоу-мо, дрон, crane shot)\n"
        "  — Активность людей (жесты, движения, эмоции на лицах)\n"
        "  — Природу/среду (частицы, блики, дым, вода, ветер, свет)\n"
        "  — Смену планов (крупный → средний → общий → деталь)\n"
        "  — Кинематографичный свет (golden hour, контровой, неон, свечи)\n"
        "  — Текст/слоган (как появляется, шрифт, анимация)\n"
        "  — Звуковую атмосферу (хотя Sora без звука, описание помогает настроению)\n"
        "  — Цветовую палитру и grade (тёплые тона, холодный blue grade, vintage)\n"
        "  На РУССКОМ.\n\n"
        "ФОРМАТ ОТВЕТА (строго):\n"
        "КАРТИНКА: [промпт]\n"
        "ВИДЕО: [промпт]"
    )

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
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 2500,
        "temperature": 0.7,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{_OPENAI_BASE}/chat/completions", data=data, method="POST")
    for k, v in _openai_headers().items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    text = body["choices"][0]["message"]["content"].strip()

    img_prompt = text
    vid_prompt = text
    if "КАРТИНКА:" in text and "ВИДЕО:" in text:
        parts = text.split("ВИДЕО:")
        img_prompt = parts[0].replace("КАРТИНКА:", "").strip()
        vid_prompt = parts[1].strip()

    return img_prompt, vid_prompt


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
        "prompt": prompt[:2500],
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

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    elif ext == "webp":
        mime = "image/webp"
    else:
        mime = "image/png"

    duration_sec = 10 if duration_sec > 5 else 5
    body_data = {
        "model_name": "kling-v2-master",
        "image": f"data:{mime};base64,{image_b64}",
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


# ── Hedra (fallback — animated portrait with TTS) ──

def _hedra_json(url: str, method: str = "GET", headers: Optional[dict] = None,
                body: Optional[bytes] = None, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, method=method, headers=headers or {}, data=body)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")[:300] if e.fp else ""
        raise RuntimeError(f"Hedra HTTP {e.code}: {err}") from e


def generate_hedra_video(image_path: str, text: str) -> str:
    """Animate image as talking portrait via Hedra + TTS."""
    if not _HEDRA_KEY:
        raise RuntimeError("HEDRA_API_KEY not configured")

    # 1. Upload image as portrait
    boundary = f"----HedraUp{int(time.time())}"
    fname = os.path.basename(image_path)
    ct = "image/jpeg" if fname.endswith(".jpg") else "image/png"
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{fname}\"\r\nContent-Type: {ct}\r\n\r\n".encode(),
        img_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    upload_resp = _hedra_json(
        f"{_HEDRA_BASE}/v1/portrait?aspect_ratio=16:9",
        method="POST",
        headers={
            "X-API-KEY": _HEDRA_KEY,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        body=body,
    )
    portrait_url = upload_resp.get("url")
    if not portrait_url:
        raise RuntimeError(f"Hedra upload failed: {upload_resp}")

    # 2. Create character video with TTS
    # Keep text short for natural speech (max ~200 chars)
    short_text = text[:200].rsplit(" ", 1)[0] if len(text) > 200 else text
    payload = {
        "audioSource": "tts",
        "text": short_text,
        "avatarImage": portrait_url,
        "aspectRatio": "16:9",
    }
    gen_resp = _hedra_json(
        f"{_HEDRA_BASE}/v1/characters",
        method="POST",
        headers={
            "X-API-KEY": _HEDRA_KEY,
            "Content-Type": "application/json",
        },
        body=json.dumps(payload).encode("utf-8"),
    )
    job_id = gen_resp.get("jobId") or gen_resp.get("project_id") or gen_resp.get("id")
    if not job_id:
        raise RuntimeError(f"Hedra generate failed: {gen_resp}")

    # 3. Poll for completion (max 3 min)
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(8)
        status = _hedra_json(
            f"{_HEDRA_BASE}/v1/projects/{job_id}",
            headers={"X-API-KEY": _HEDRA_KEY},
        )
        state = status.get("status") or status.get("stage") or "unknown"
        if state in ("Completed", "completed", "done", "success"):
            video_url = status.get("videoUrl") or status.get("video_url") or status.get("url")
            if not video_url:
                raise RuntimeError("Hedra: completed but no video URL")
            # Download
            with urllib.request.urlopen(video_url, timeout=120) as resp:
                vbytes = resp.read()
            fpath = os.path.join(_OUTPUT_DIR, f"hedra_{job_id}.mp4")
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
        if state in ("Failed", "failed", "error"):
            raise RuntimeError(f"Hedra failed: {status.get('errorMessage') or status}")
    raise RuntimeError("Hedra: таймаут (3 мин)")


# ── Background job ──

def _run_job(job_id: str, idea: str, ref_images: list[tuple[bytes, str]]):
    job = _jobs[job_id]
    q: queue.Queue = job["queue"]

    def emit(msg: str, **extra):
        event = {"msg": msg, **extra}
        q.put(event)

    try:
        # 1. GPT
        emit("GPT создаёт креативный бриф...", step="gpt")
        img_prompt, vid_prompt = gpt_creative_prompts(idea, ref_images if ref_images else None)
        emit(f"Бриф готов", step="gpt_done", img_prompt=img_prompt, vid_prompt=vid_prompt)

        # 2. DALL-E
        emit("Генерирую картинку (DALL-E 3 HD)...", step="dalle")
        img_path = generate_dalle(img_prompt)
        img_name = os.path.basename(img_path)
        emit("Картинка готова!", step="dalle_done", image=f"/output/{img_name}")

        # 3. Resize
        video_img = resize_for_video(img_path, 1280, 720)

        # 4. Video (Kling primary → Sora fallback)
        vid_path = None
        vid_label = ""
        errors = []
        # Try Kling first (more reliable)
        if _KLING_ACCESS and _KLING_SECRET:
            try:
                emit("Генерирую видео (Kling, ~1-3 мин)...", step="video")
                vid_path = generate_kling_video(video_img, vid_prompt, duration_sec=10)
                vid_label = "Kling"
            except Exception as kling_err:
                errors.append(f"Kling: {kling_err}")
                emit(f"Kling: {str(kling_err)[:120]}. Пробую Sora...", step="video_fallback")
        # Fallback to Sora
        if not vid_path:
            try:
                emit("Генерирую видео (Sora 2)...", step="video")
                vid_path = generate_sora_video(video_img, vid_prompt, seconds=8)
                vid_label = "Sora 2"
            except Exception as sora_err:
                errors.append(f"Sora: {sora_err}")
                emit(f"Sora: {str(sora_err)[:120]}. Пробую Hedra...", step="video_fallback")
        # Fallback to Hedra (animated portrait + TTS)
        if not vid_path and _HEDRA_KEY:
            try:
                emit("Генерирую видео (Hedra)...", step="video")
                vid_path = generate_hedra_video(video_img, idea)
                vid_label = "Hedra"
            except Exception as hedra_err:
                errors.append(f"Hedra: {hedra_err}")
        if not vid_path:
            raise RuntimeError("Видео не удалось:\n" + "\n".join(errors))

        vid_name = os.path.basename(vid_path)
        emit("Готово!", step="done", video=f"/output/{vid_name}", video_provider=vid_label)
        job["status"] = "done"

        # Save to shared history
        _save_history_entry({
            "idea": idea,
            "img_prompt": img_prompt,
            "vid_prompt": vid_prompt,
            "image": f"/output/{img_name}",
            "video": f"/output/{vid_name}",
            "provider": vid_label,
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
    if not _OPENAI_KEY:
        return jsonify({"error": "OPENAI_API_KEY not configured"}), 500

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
