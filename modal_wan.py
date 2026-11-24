"""
Antuan Music · portadas con vídeo IA real (Wan 2.2) en una GPU de Modal, sin el PC.

- `tick` (CPU, cada 2 min, casi gratis) mira Supabase. Si hay portadas encargadas o portadas que aún no
  tienen el vídeo Wan nuevo (prefijo "w5-"), despierta `render` (GPU).
- `render` carga Wan 2.2 TI2V-5B una vez y hace portadas hasta ~14 min; luego se apaga (sin portadas no gasta).
  Orden: primero las encargadas, luego las del sistema antiguo y por último las IA de Hugging Face ("ai-").
  Las "ai-" no se borran: se mueven a loops/backup/ por si hay que volver a ellas.
- Cada vídeo se sube una vez a loops/ y se queda para siempre (solo se borra si el usuario lo pide).

Despliegue: GitHub Actions → "Modal Wan" (usa los secretos MODAL_TOKEN_ID y MODAL_TOKEN_SECRET).
Prueba a mano: modal run modal_wan.py::test --url <imagen>
"""
import os
import time

import modal

app = modal.App("antuan-wan")
MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
CAPTIONER = "Salesforce/blip-image-captioning-large"
cache = modal.Volume.from_name("antuan-hf-cache", create_if_missing=True)
lock = modal.Dict.from_name("antuan-wan-lock", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.7.1", "torchvision==0.22.1", "transformers>=4.51,<5", "accelerate", "ftfy",
        "sentencepiece", "safetensors", "imageio", "imageio-ffmpeg", "pillow", "numpy<2", "requests",
        "diffusers==0.35.2", "huggingface_hub",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
)
cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("requests")
secrets = [modal.Secret.from_name("antuan-supabase")]

GPU = os.environ.get("WAN_GPU", "A10G")
STEPS = int(os.environ.get("WAN_STEPS", "30"))
FRAMES = 49          # 4k+1 frames → ~2 s at 24 fps; the loop is forward + backward (~4 s)
SIDE = 704           # square covers; Wan 2.2 5B wants multiples of 32
OUT = 512            # same size as the old loops on the phones


# ---------- the prompt (same rules as animate_worker.cover_prompt) ----------
_PEOPLE = (" man ", " woman ", " men ", " women ", " person ", " people ", " boy ", " girl ", " guy ", " face ",
           " singer ", " rapper ", " child ", " kid ", " couple ", " his ", " her ")
_MOTIONS = [
    ((" rain", " raining", " storm"), "the rain keeps falling"),
    ((" snow",), "snowflakes keep falling"),
    ((" neon", " lights", " lamp", " sign", " city at night", " street at night"), "the lights flicker gently"),
    ((" fire", " flame", " candle"), "the flames flicker"),
    ((" smoke", " fog", " mist", " cloud"), "the smoke and clouds drift slowly"),
    ((" water", " sea", " ocean", " river", " pool", " lake", " wave"), "the water ripples"),
    ((" tree", " palm", " leaves", " plant", " garden", " grass", " forest", " jungle", " flower", " field"),
     "leaves and grass sway softly in a light breeze"),
    ((" horse", " dog", " cat ", " bird", " animal", " bull", " lion", " tiger", " wolf"),
     "the animal moves naturally in place, its head, mane, tail and body moving"),
    ((" hair",), "the hair moves in the wind"),
    ((" dress", " shirt", " jacket", " flag", " curtain"), "the fabric moves slightly in the wind"),
    ((" car ", " road", " highway"), "light reflections slide over the surfaces"),
    ((" sky", " sunset", " sun "), "the clouds in the sky drift slowly"),
]
NEG = ("camera movement, zoom, pan, dolly, camera shake, cuts, scene change, new objects, new people, rain added, "
       "face change, identity change, different face, face morphing, deformed face, warped eyes, extra limbs, "
       "extra fingers, melting, morphing background, text changes, morphing letters, new text, logo changes, "
       "style change, color shift, sharpening blurry areas, refocusing, depth of field change, flicker, "
       "watermark, static image, frozen, low quality, artifacts")


def cover_prompt(caption: str) -> str:
    c = f" {caption.lower()} "
    moves = []
    for words, move in _MOTIONS:
        if any(w in c for w in words) and move not in moves:
            moves.append(move)
    people = any(w in c for w in _PEOPLE)
    if not moves and not people:
        moves.append("the light shifts very softly")
    motion = ", ".join(moves[:4])
    return (
        f"{caption}. A subtle living version of this exact photo"
        + (f": {motion}" if motion else "")
        + (". The person breathes, blinks and moves naturally and slightly, with the same face, identity and "
           "expression" if people else "")
        + ". Nothing else is added: no rain, snow, smoke, particles, people or objects that are not already in the "
          "photo. The camera is completely still, same framing, no zoom, no pan. Blurry or motion-blurred areas stay "
          "exactly as blurry, never sharpened or refocused. Same colors, lighting, grain and style; all text and "
          "logos unchanged. Calm, smooth motion that can loop."
    )


# ---------- Supabase ----------
def _sb():
    import requests

    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_KEY"]

    def H(extra=None):
        h = {"apikey": key}
        if not key.startswith("sb_"):
            h["Authorization"] = f"Bearer {key}"
        h.update(extra or {})
        return h

    def rest(method, path, **kw):
        r = requests.request(method, f"{url}/rest/v1/{path}", timeout=30, **kw)
        r.raise_for_status()
        return r.json() if r.content else None

    return url, H, rest


def _next_job():
    """Queued covers first, then old loops, then old Hugging Face videos. Returns a row or None."""
    url, H, rest = _sb()
    rows = rest("GET", "cover_loops?status=eq.queued&order=requested_at.asc&limit=1&select=cover_key,cover_url,video_path",
                headers=H())
    if rows:
        got = rest("PATCH", f"cover_loops?cover_key=eq.{rows[0]['cover_key']}&status=eq.queued",
                   headers=H({"Prefer": "return=representation", "Content-Type": "application/json"}),
                   json={"status": "processing", "updated_at": "now()"})
        if got:
            return got[0]
    for flt in ("video_path=not.like.ai-*&video_path=not.like.w5-*", "video_path=like.ai-*"):
        rows = rest("GET", f"cover_loops?status=eq.done&{flt}&order=requested_at.desc&limit=1"
                           "&select=cover_key,cover_url,video_path", headers=H())
        if rows:
            return rows[0]
    return None


def _pending() -> bool:
    url, H, rest = _sb()
    for flt in ("status=in.(queued,processing)", "status=eq.done&video_path=not.like.w5-*"):
        if rest("GET", f"cover_loops?{flt}&select=cover_key&limit=1", headers=H()):
            return True
    return False


def _save(row: dict, data: bytes):
    import requests

    url, H, rest = _sb()
    key, old = row["cover_key"], row.get("video_path")
    path = f"w5-{key}-{int(time.time())}.mp4"
    r = requests.post(f"{url}/storage/v1/object/loops/{path}", data=data, timeout=120,
                      headers=H({"Content-Type": "video/mp4", "x-upsert": "true", "cache-control": "31536000"}))
    r.raise_for_status()
    rest("PATCH", f"cover_loops?cover_key=eq.{key}", headers=H({"Content-Type": "application/json"}),
         json={"status": "done", "video_path": path, "error": None, "updated_at": "now()"})
    if old and old != path:
        if old.startswith("ai-"):  # keep the Hugging Face videos, out of the way
            requests.post(f"{url}/storage/v1/object/move", timeout=30, headers=H({"Content-Type": "application/json"}),
                          json={"bucketId": "loops", "sourceKey": old, "destinationKey": f"backup/{old}"})
        else:
            requests.delete(f"{url}/storage/v1/object/loops/{old}", headers=H(), timeout=30)
    return path


def _fail(row: dict, err: str):
    url, H, rest = _sb()
    if row.get("video_path"):  # it already had a working loop: leave it as it was, just log
        print("   fallo, se queda el bucle que tenía:", err[:200])
        return
    rest("PATCH", f"cover_loops?cover_key=eq.{row['cover_key']}", headers=H({"Content-Type": "application/json"}),
         json={"status": "failed", "error": err[:500], "updated_at": "now()"})


# ---------- GPU ----------
@app.cls(image=gpu_image, gpu=GPU, volumes={"/cache": cache}, secrets=secrets, timeout=20 * 60,
         max_containers=1, scaledown_window=60)
class Wan:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from transformers import BlipForConditionalGeneration, BlipProcessor

        t = time.time()
        vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
        self.pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16,
                                                            expand_timesteps=True)
        self.pipe.enable_model_cpu_offload()  # 24 GB GPU: text encoder and transformer take turns
        self.cap_proc = BlipProcessor.from_pretrained(CAPTIONER)
        self.cap = BlipForConditionalGeneration.from_pretrained(CAPTIONER, torch_dtype=torch.float16).to("cuda").eval()
        cache.commit()
        print(f"[modelo] Wan 2.2 listo en {time.time() - t:.0f}s")

    def _caption(self, img) -> str:
        import torch

        with torch.no_grad():
            inp = self.cap_proc(images=img, return_tensors="pt").to("cuda", torch.float16)
            out = self.cap.generate(**inp, max_new_tokens=40)
        txt = self.cap_proc.decode(out[0], skip_special_tokens=True).strip()
        return txt.removeprefix("arafed ").removeprefix("araffe ") or "an album cover"

    def _video(self, url: str) -> tuple[bytes, str, float]:
        import io
        import tempfile

        import imageio
        import numpy as np
        import requests
        import torch
        from PIL import Image

        t = time.time()
        img = Image.open(io.BytesIO(requests.get(url, timeout=30).content)).convert("RGB")
        img = img.resize((SIDE, SIDE), Image.LANCZOS)  # covers are square
        caption = self._caption(img)
        prompt = cover_prompt(caption)
        print("   descripción:", caption)
        with torch.inference_mode():
            frames = self.pipe(image=img, prompt=prompt, negative_prompt=NEG, height=SIDE, width=SIDE,
                               num_frames=FRAMES, num_inference_steps=STEPS, guidance_scale=5.0,
                               generator=torch.Generator("cuda").manual_seed(7), output_type="np").frames[0]
        frames = [np.asarray(Image.fromarray((f * 255).clip(0, 255).astype("uint8")).resize((OUT, OUT), Image.LANCZOS))
                  for f in frames]
        loop = frames + frames[-2:0:-1]  # forward then backward: seamless, no jump
        tmp = tempfile.mktemp(suffix=".mp4")
        with imageio.get_writer(tmp, fps=24, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=16,
                                ffmpeg_params=["-movflags", "+faststart"]) as wr:
            for f in loop:
                wr.append_data(f)
        data = open(tmp, "rb").read()
        os.remove(tmp)
        return data, caption, time.time() - t

    @modal.method()
    def drain(self, max_seconds: float = 14 * 60) -> int:
        start, done = time.time(), 0
        try:
            while time.time() - start < max_seconds:
                lock.put("busy", time.time())
                row = _next_job()
                if not row:
                    break
                try:
                    data, caption, secs = self._video(row["cover_url"])
                    path = _save(row, data)
                    done += 1
                    print(f"[ok] {secs:.0f}s  {path}")
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    _fail(row, f"{type(e).__name__}: {e}")
        finally:
            lock.pop("busy", None)
        return done

    @modal.method()
    def one(self, url: str) -> tuple[bytes, str, float]:
        return self._video(url)


@app.function(image=cpu_image, secrets=secrets, schedule=modal.Period(minutes=2), timeout=60)
def tick():
    busy = lock.get("busy", 0)
    if busy and time.time() - busy < 20 * 60:
        return "ocupado"
    if not _pending():
        return "nada"
    lock.put("busy", time.time())
    Wan().drain.spawn()
    return "despertado"


@app.local_entrypoint()
def test(url: str = "https://is1-ssl.mzstatic.com/image/thumb/Music221/v4/90/5e/7e/905e7ed5-a8fa-a8f3-cd06-0028fdf3afaa/199066342442.jpg/1000x1000bb.jpg",
         out: str = "prueba_wan.mp4"):
    t = time.time()
    data, caption, secs = Wan().one.remote(url)
    open(out, "wb").write(data)
    print(f"descripción: {caption}")
    print(f"vídeo en {secs:.0f}s de GPU ({time.time() - t:.0f}s contando el arranque) → {out}")
