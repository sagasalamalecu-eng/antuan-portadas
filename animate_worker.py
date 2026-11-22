"""
# -*- coding: utf-8 -*-
Antuan Music · AI cover-loop worker (runs on the PC with the GPU).

Loop:  take a queued cover from Supabase  →  animate the face with LivePortrait
       (head sway, breathing, one blink; every curve is periodic → seamless loop)
       →  encode a small MP4  →  upload to the "loops" bucket  →  mark it done.

Config (D:/AntuanAI/.env or environment):
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY=...      # service_role key — keep it ONLY on this PC
  LIVEPORTRAIT_DIR=D:/AntuanAI/LivePortrait   (optional)

Usage:
  python animate_worker.py                 # run forever, polling the queue
  python animate_worker.py --test img.jpg  # just render one local image to out.mp4
"""
import io
import math
import tempfile
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


# ---------- config ----------
def load_env(path: Path):
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


load_env(Path("D:/AntuanAI/.env"))
load_env(Path(__file__).with_name(".env"))

LP_DIR = Path(os.environ.get("LIVEPORTRAIT_DIR", "D:/AntuanAI/LivePortrait"))
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

FPS = int(os.environ.get("LOOP_FPS", "30"))
SECONDS = float(os.environ.get("LOOP_SECONDS", "4.0"))  # loop length (FPS*SECONDS frames)
SIZE = 512
FACE_KEYFRAMES = int(os.environ.get("FACE_KEYFRAMES", "0"))  # >0: fast mode for CPU servers (output video is SIZE×SIZE)
POLL_SECONDS = 4

sys.path.insert(0, str(LP_DIR))
os.chdir(LP_DIR)  # LivePortrait resolves some resources relative to cwd

import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from src.config.inference_config import InferenceConfig  # noqa: E402
from src.config.crop_config import CropConfig  # noqa: E402
from src.live_portrait_pipeline import LivePortraitPipeline  # noqa: E402
from src.utils.camera import get_rotation_matrix  # noqa: E402
from src.utils.crop import prepare_paste_back, paste_back  # noqa: E402
from src.utils.retargeting_utils import calc_eye_close_ratio  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import scene_fx  # noqa: E402


# ---------- model ----------
def build_pipeline():
    inf = InferenceConfig()
    inf.flag_use_half_precision = False  # GTX 10xx (Pascal) is very slow in fp16
    inf.flag_force_cpu = not torch.cuda.is_available()
    return LivePortraitPipeline(inference_cfg=inf, crop_cfg=CropConfig())


PIPE = None


def pipe():
    global PIPE
    if PIPE is None:
        t = time.time()
        PIPE = build_pipeline()
        print(f"[model] LivePortrait ready in {time.time() - t:.1f}s on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    return PIPE


# ---------- animation ----------
class NoFace(Exception):
    pass


@torch.no_grad()
def animate(img_rgb: np.ndarray) -> list:
    """Returns a list of RGB frames forming a seamless loop."""
    p = pipe()
    w = p.live_portrait_wrapper
    crop = p.cropper.crop_source_image(img_rgb, p.cropper.crop_cfg)
    if crop is None:
        raise NoFace()

    I_s = w.prepare_source(crop["img_crop_256x256"])
    lmk = crop["lmk_crop"]
    M_c2o = crop["M_c2o"]
    mask = prepare_paste_back(w.inference_cfg.mask_crop, M_c2o, dsize=(img_rgb.shape[1], img_rgb.shape[0]))

    info = w.get_kp_info(I_s)
    f_s = w.extract_feature_3d(I_s)
    x_s = w.transform_keypoint(info)
    kp, exp, scale, tr = info["kp"], info["exp"], info["scale"], info["t"]
    pitch0, yaw0, roll0 = info["pitch"], info["yaw"], info["roll"]

    eye0 = float(np.asarray(calc_eye_close_ratio(lmk[None])).mean())
    can_blink = eye0 > 0.18  # don't "blink" closed eyes / sunglasses-ish crops

    def render(ph: float) -> np.ndarray:
        """One frame of the loop. Every curve is a function of cos(2πph) → symmetric: f(ph) == f(1-ph)."""
        c = math.cos(2 * math.pi * ph)
        c2 = math.cos(4 * math.pi * ph)
        # subtle on purpose: bigger values look like a bobble-head
        d_yaw = 2.2 * c                  # degrees: look left ↔ right and back
        d_pitch = 0.8 * c2               # slight nod
        d_roll = 0.7 * c                 # head tilt follows the turn
        breathe = 1.0 + 0.01 * (0.5 - 0.5 * c)
        R_d = get_rotation_matrix(pitch0 + d_pitch, yaw0 + d_yaw, roll0 + d_roll)
        x_d = scale * breathe * (kp @ R_d + exp) + tr
        if can_blink:
            blink = math.exp(-(((ph - 0.5) / 0.035) ** 2))  # one quick blink, centred → symmetric
            if blink > 0.02:
                comb = w.calc_combined_eye_ratio([[eye0 * (1 - 0.92 * blink)]], lmk)
                x_d = x_d + w.retarget_eye(x_s, comb)
        x_d = w.stitching(x_s, x_d)
        out = w.warp_decode(f_s, x_s, x_d)
        face = w.parse_output(out["out"])[0]
        return paste_back(face, M_c2o, img_rgb, mask)

    n = int(FPS * SECONDS)
    n -= n % 2
    half = n // 2
    # second half of the loop is the first half reversed (motion is symmetric) → render only half
    if FACE_KEYFRAMES > 0:
        # fast mode (CPU servers): AI-render a few keyframes, fill the rest with optical-flow interpolation
        step = max(1, half // FACE_KEYFRAMES)
        keys = sorted(set(list(range(0, half, step)) + [half - 3, half - 1, half] if can_blink else list(range(0, half, step)) + [half]))
        rendered = {i: render(i / n) for i in keys}
        first = [None] * (half + 1)
        for a, b in zip(keys, keys[1:]):
            first[a] = rendered[a]
            for i in range(a + 1, b):
                first[i] = _interp(rendered[a], rendered[b], (i - a) / (b - a))
        first[half] = rendered[half]
    else:
        first = [render(i / n) for i in range(half + 1)]
    return first[:half] + [first[n - i] for i in range(half, n)]


_DIS = None


def _interp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Intermediate frame between two close frames using dense optical flow (fast on CPU)."""
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    flow = _DIS.calc(ga, gb, None)  # a → b
    h, w = ga.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wa = cv2.remap(a, gx - t * flow[..., 0], gy - t * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(b, gx + (1 - t) * flow[..., 0], gy + (1 - t) * flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.addWeighted(wa, 1 - t, wb, t, 0)


def make_loop(img: np.ndarray) -> list:
    """Face animation (if there is a face) + scene-aware effects for everything else."""
    t = time.time()
    try:
        frames = animate(img)
        face = True
    except NoFace:
        frames = [img] * int(FPS * SECONDS)
        face = False
    t_face = time.time() - t
    info = scene_fx.analyze(img)
    frames = scene_fx.apply(frames, img, info)
    print(f"   cara={'si' if face else 'no'} ({t_face:.0f}s) {scene_fx.describe(info)} total {time.time() - t:.0f}s")
    return frames


def to_mp4(frames: list) -> bytes:
    buf = io.BytesIO()
    tmp = Path(tempfile.gettempdir()) / f"loop_{os.getpid()}.mp4"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(tmp, fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p",
                            macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"]) as wr:
        for f in frames:
            wr.append_data(np.asarray(Image.fromarray(f).resize((SIZE, SIZE), Image.LANCZOS)))
    data = tmp.read_bytes()
    tmp.unlink(missing_ok=True)
    buf.write(data)
    return buf.getvalue()


def load_image(url_or_path: str) -> np.ndarray:
    if url_or_path.startswith("http"):
        r = requests.get(url_or_path, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
    else:
        img = Image.open(url_or_path)
    img = img.convert("RGB")
    img.thumbnail((768, 768))
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    return arr[: h - h % 2, : w - w % 2].copy()


# ---------- Supabase ----------
def H(extra=None):
    # new-style secret keys (sb_secret_...) go only in "apikey"; legacy service_role JWTs also as Bearer
    h = {"apikey": SERVICE_KEY}
    if not SERVICE_KEY.startswith("sb_"):
        h["Authorization"] = f"Bearer {SERVICE_KEY}"
    h.update(extra or {})
    return h


def rest(method, path, **kw):
    r = requests.request(method, f"{SUPABASE_URL}/rest/v1/{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def claim():
    rows = rest("GET", "cover_loops?status=eq.queued&order=requested_at.asc&limit=1&select=cover_key,cover_url", headers=H())
    if not rows:
        return None
    key = rows[0]["cover_key"]
    got = rest("PATCH", f"cover_loops?cover_key=eq.{key}&status=eq.queued",
               headers=H({"Prefer": "return=representation", "Content-Type": "application/json"}),
               json={"status": "processing", "updated_at": "now()"})
    return got[0] if got else None


def finish(key, **fields):
    fields["updated_at"] = "now()"
    rest("PATCH", f"cover_loops?cover_key=eq.{key}", headers=H({"Content-Type": "application/json"}), json=fields)


def upload(key, data: bytes) -> str:
    path = f"{key}.mp4"
    r = requests.post(f"{SUPABASE_URL}/storage/v1/object/loops/{path}", data=data, timeout=120,
                      headers=H({"Content-Type": "video/mp4", "x-upsert": "true", "cache-control": "31536000"}))
    r.raise_for_status()
    return path


def requeue_stale():
    # jobs left "processing" by a crash/shutdown go back to the queue
    rest("PATCH", "cover_loops?status=eq.processing", headers=H({"Content-Type": "application/json"}),
         json={"status": "queued"})


def drain(max_seconds: float = 600) -> int:
    """Process queued covers until the queue is empty or time runs out (used by the cloud server)."""
    pipe()
    requeue_stale()
    start, done = time.time(), 0
    while time.time() - start < max_seconds:
        job = claim()
        if not job:
            break
        key, url = job["cover_key"], job["cover_url"]
        try:
            frames = make_loop(load_image(url))
            finish(key, status="done", video_path=upload(key, to_mp4(frames)), error=None)
            done += 1
            print(f"[ok] {url[:70]}")
        except Exception as e:
            traceback.print_exc()
            finish(key, status="failed", error=str(e)[:500])
    return done


def run():
    if not SUPABASE_URL or not SERVICE_KEY:
        sys.exit("Falta SUPABASE_URL o SUPABASE_SERVICE_KEY en D:/AntuanAI/.env")
    pipe()
    requeue_stale()
    print("[worker] esperando portadas...")
    while True:
        try:
            job = claim()
        except Exception as e:
            print("[worker] supabase no responde:", e)
            time.sleep(15)
            continue
        if not job:
            time.sleep(POLL_SECONDS)
            continue
        key, url = job["cover_key"], job["cover_url"]
        t = time.time()
        try:
            frames = make_loop(load_image(url))
            path = upload(key, to_mp4(frames))
            finish(key, status="done", video_path=path, error=None)
            print(f"[ok] {url[:70]} ({time.time() - t:.1f}s)")
        except Exception as e:
            traceback.print_exc()
            finish(key, status="failed", error=str(e)[:500])


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--test":
        t = time.time()
        frames = make_loop(load_image(sys.argv[2]))
        out = Path(sys.argv[3] if len(sys.argv) > 3 else "D:/AntuanAI/out.mp4")
        out.write_bytes(to_mp4(frames))
        print(f"{len(frames)} frames -> {out} in {time.time() - t:.1f}s")
    else:
        run()
