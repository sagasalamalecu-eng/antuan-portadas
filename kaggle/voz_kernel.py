# Antuan Music · "Quitar la voz": separates every song into beat + voice on a free Kaggle GPU.
# kaggle_voz.py fills JOBS (video id + signed URL of the song in Supabase) before each push.
# Output: /kaggle/working/stems/<id>.inst.m4a and <id>.vocals.m4a + results.json.
JOBS = []
BATCH = ""
BUDGET_S = 6000

import json
import os
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "audio-separator[gpu]"], check=True)

import requests  # noqa: E402
from audio_separator.separator import Separator  # noqa: E402

# BS-RoFormer: the best open model at removing the voice (much cleaner beat than Demucs/Spleeter)
MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
OUT = Path("/kaggle/working/stems")
WORK = Path("/kaggle/working/tmp")
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
results = {"batch": BATCH, "done": [], "failed": [], "seconds": {}}


def save_results():
    json.dump(results, open("/kaggle/working/results.json", "w"))


save_results()
sep = Separator(output_dir=str(WORK), output_format="WAV")
sep.load_model(model_filename=MODEL)
print(f"[voz] modelo listo en {time.time() - T0:.0f}s · {len(JOBS)} canciones")


def to_m4a(src: str, dst: Path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(dst)], check=True)


for job in JOBS:
    if time.time() - T0 > BUDGET_S:
        print("[voz] tiempo de esta tanda agotado")
        break
    vid = job["id"]
    t = time.time()
    try:
        song = WORK / f"{vid}.m4a"
        song.write_bytes(requests.get(job["url"], timeout=120).content)
        files = [str(WORK / f) if not os.path.isabs(f) else f for f in sep.separate(str(song))]
        voc = next(f for f in files if "(Vocals)" in f)
        ins = next(f for f in files if "(Instrumental)" in f)
        to_m4a(ins, OUT / f"{vid}.inst.m4a")
        to_m4a(voc, OUT / f"{vid}.vocals.m4a")
        for f in [*files, str(song)]:
            Path(f).unlink(missing_ok=True)
        results["done"].append(vid)
        results["seconds"][vid] = round(time.time() - t, 1)
        print(f"[ok] {time.time() - t:.0f}s  {vid}")
    except Exception as e:
        results["failed"].append({"id": vid, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        print(f"[fallo] {vid}: {type(e).__name__}: {e}")
    save_results()

print(f"[voz] hechas {len(results['done'])} en {time.time() - T0:.0f}s")
