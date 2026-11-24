# Antuan Music · runs on a free Kaggle GPU (T4). kaggle_bridge.py writes the JOBS line below before each push.
# For every cover: caption it (BLIP), build the prompt, animate it with Wan 2.2 TI2V-5B and save
# /kaggle/working/out/<cover_key>.mp4 + results.json (the bridge uploads them to Supabase afterwards).
JOBS = []  # filled in by kaggle_bridge.py
BATCH = ""
BUDGET_S = 6000

import io
import json
import os
import subprocess
import sys
import time

T0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.35.2", "transformers>=4.51,<5",
                "accelerate", "ftfy", "imageio-ffmpeg", "sentencepiece"], check=True)

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import BlipForConditionalGeneration, BlipProcessor  # noqa: E402

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
SIDE, FRAMES, STEPS, OUT = 512, 49, 20, 512
OUTDIR = "/kaggle/working/out"
os.makedirs(OUTDIR, exist_ok=True)

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


results = {"batch": BATCH, "done": [], "failed": [], "seconds": {}}


def save_results():
    json.dump(results, open("/kaggle/working/results.json", "w"))


save_results()
print(f"[kaggle] {len(JOBS)} portadas · GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA'}")

cap_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
cap = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large",
                                                   torch_dtype=torch.float16).to("cuda").eval()


def load_pipe(dtype):
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    p = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=dtype, expand_timesteps=True)
    p.enable_model_cpu_offload()  # T4 has 16 GB: text encoder and transformer take turns on the GPU
    return p


DTYPE = torch.float16  # T4 has no fast bfloat16; if fp16 gives NaN/black frames we switch once to bf16
t = time.time()
pipe = load_pipe(DTYPE)
print(f"[kaggle] Wan 2.2 cargado en {time.time() - t:.0f}s")


def caption_of(img) -> str:
    with torch.no_grad():
        inp = cap_proc(images=img, return_tensors="pt").to("cuda", torch.float16)
        out = cap.generate(**inp, max_new_tokens=40)
    txt = cap_proc.decode(out[0], skip_special_tokens=True).strip()
    for junk in ("arafed ", "araffe ", "there is "):
        txt = txt.removeprefix(junk)
    return txt or "an album cover"


def render(img, prompt):
    with torch.inference_mode():
        fr = pipe(image=img, prompt=prompt, negative_prompt=NEG, height=SIDE, width=SIDE, num_frames=FRAMES,
                  num_inference_steps=STEPS, guidance_scale=5.0, generator=torch.Generator("cpu").manual_seed(7),
                  output_type="np").frames[0]
    return np.nan_to_num(np.asarray(fr, dtype=np.float32))


for job in JOBS:
    if time.time() - T0 > BUDGET_S:
        print("[kaggle] tiempo de esta tanda agotado, el resto en la siguiente")
        break
    key = job["key"]
    t = time.time()
    try:
        img = Image.open(io.BytesIO(requests.get(job["url"], timeout=30).content)).convert("RGB")
        img = img.resize((SIDE, SIDE), Image.LANCZOS)
        caption = caption_of(img)
        prompt = cover_prompt(caption)
        fr = render(img, prompt)
        if fr.max() < 0.02 and DTYPE == torch.float16:  # fp16 overflow → black video: retry in bf16
            print("   fp16 dio vídeo negro, paso a bf16")
            del pipe
            torch.cuda.empty_cache()
            DTYPE = torch.bfloat16
            pipe = load_pipe(DTYPE)
            fr = render(img, prompt)
        frames = [np.asarray(Image.fromarray((f * 255).clip(0, 255).astype("uint8")).resize((OUT, OUT), Image.LANCZOS))
                  for f in fr]
        loop = frames + frames[-2:0:-1]  # forward then backward: seamless
        with imageio.get_writer(f"{OUTDIR}/{key}.mp4", fps=24, codec="libx264", quality=7, pixelformat="yuv420p",
                                macro_block_size=16, ffmpeg_params=["-movflags", "+faststart"]) as wr:
            for f in loop:
                wr.append_data(f)
        secs = time.time() - t
        results["done"].append({"key": key, "caption": caption})
        results["seconds"][key] = round(secs, 1)
        print(f"[ok] {secs:.0f}s  {key}  ({caption})")
    except Exception as e:
        results["failed"].append({"key": key, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(f"[fallo] {key}: {type(e).__name__}: {e}")
    save_results()

print(f"[kaggle] hechas {len(results['done'])} en {time.time() - T0:.0f}s")
