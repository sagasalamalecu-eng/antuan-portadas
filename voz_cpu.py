"""
Antuan Music · "Quitar la voz" en GitHub Actions (CPU, sin el PC y sin esperar a Kaggle).
Para cada canción guardada que aún no tiene sus pistas, crea songs/{id}.inst.m4a (beat) y
songs/{id}.vocals.m4a (voz) con UVR MDX-Net Inst HQ 3 (rápido en CPU, beat limpio).
Si Kaggle (kaggle_voz.py, BS-RoFormer en GPU) llega antes a una canción, aquí se salta.
"""
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import requests

import kaggle_voz as kv
import kaggle_bridge as kb

MODEL = os.environ.get("VOZ_MODEL", "UVR-MDX-NET-Inst_HQ_3.onnx")
BUDGET_S = float(os.environ.get("VOZ_BUDGET_S", str(20 * 60)))
WORK = Path("voz_tmp")


def to_m4a(src: str, dst: Path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart", str(dst)], check=True)


def main():
    t0 = time.time()
    jobs = kv.next_jobs()
    print(f"canciones sin karaoke: {len(jobs)}")
    if not jobs:
        return
    from audio_separator.separator import Separator

    WORK.mkdir(exist_ok=True)
    sep = Separator(output_dir=str(WORK), output_format="WAV", model_file_dir=os.path.expanduser("~/voz_models"))
    sep.load_model(model_filename=MODEL)
    print(f"modelo listo en {time.time() - t0:.0f}s")
    done = 0
    for job in jobs:
        if time.time() - t0 > BUDGET_S:
            print("tiempo de esta pasada agotado; el resto en la siguiente")
            break
        vid = job["id"]
        if vid in kv.stems_done():
            continue
        t = time.time()
        try:
            song = WORK / f"{vid}.m4a"
            song.write_bytes(requests.get(job["url"], timeout=120).content)
            files = [f if os.path.isabs(f) else str(WORK / f) for f in sep.separate(str(song))]
            ins = next(f for f in files if "(Instrumental)" in f)
            voc = next(f for f in files if "(Vocals)" in f)
            added = 0
            for kind, src in (("inst", ins), ("vocals", voc)):
                out = WORK / f"{vid}.{kind}.m4a"
                to_m4a(src, out)
                data = out.read_bytes()
                requests.post(f"{kb.URL}/storage/v1/object/songs/{vid}.{kind}.m4a", data=data, timeout=300,
                              headers=kb.H({"Content-Type": "audio/mp4", "x-upsert": "true"})).raise_for_status()
                added += len(data)
            row = kb.rest("GET", f"song_files?video_id=eq.{vid}&select=bytes", headers=kb.H())
            if row:
                kb.rest("PATCH", f"song_files?video_id=eq.{vid}", headers=kb.H({"Content-Type": "application/json"}),
                        json={"bytes": (row[0].get("bytes") or 0) + added})
            for f in WORK.iterdir():
                f.unlink(missing_ok=True)
            done += 1
            print(f"[ok] {time.time() - t:.0f}s  {vid}")
        except Exception:
            traceback.print_exc()
    print(f"karaoke listo en {done} canciones ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
