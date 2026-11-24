"""
Antuan Music · puente "Quitar la voz" (GitHub Actions ⇄ Kaggle), igual que kaggle_bridge.py para las portadas.
Para cada canción guardada en Supabase (songs/{id}.m4a) crea songs/{id}.inst.m4a (solo beat) y
songs/{id}.vocals.m4a (solo voz). La app las reproduce a la vez y el slider solo cambia el volumen de la voz.
"""
import json
import shutil
import tempfile
import time
from pathlib import Path

import requests

import kaggle_bridge as kb

SLUG = "antuan-voz"
STATE_OBJ = "worker/kaggle_voz_state.json"
BATCH_SIZE = 25


def stems_done() -> set:
    r = requests.post(f"{kb.URL}/storage/v1/object/list/songs", timeout=30,
                      headers=kb.H({"Content-Type": "application/json"}),
                      json={"prefix": "", "limit": 10000, "offset": 0})
    r.raise_for_status()
    return {o["name"][:-9] for o in r.json() if o["name"].endswith(".inst.m4a")}


def next_jobs() -> list:
    have = stems_done()
    rows = kb.rest("GET", "song_files?status=eq.done&path=not.is.null&order=last_played.desc&limit=200"
                          "&select=video_id,path", headers=kb.H()) or []
    jobs = []
    for r in rows:
        if r["video_id"] in have:
            continue
        s = requests.post(f"{kb.URL}/storage/v1/object/sign/songs/{r['path']}", timeout=30,
                          headers=kb.H({"Content-Type": "application/json"}), json={"expiresIn": 6 * 3600})
        if s.ok:
            jobs.append({"id": r["video_id"], "url": f"{kb.URL}/storage/v1{s.json()['signedURL']}"})
        if len(jobs) >= BATCH_SIZE:
            break
    return jobs


def collect(user: str, state: dict):
    tmp = Path(tempfile.mkdtemp())
    kb.kaggle("kernels", "output", f"{user}/{SLUG}", "-p", str(tmp), "-o")
    res_file = next(tmp.rglob("results.json"), None)
    if not res_file:
        print("sin results.json")
        return
    res = json.loads(res_file.read_text())
    if res.get("batch") != state.get("batch"):
        print("la salida es de otra tanda")
        return
    for vid in res.get("done", []):
        added = 0
        for kind in ("inst", "vocals"):
            f = next(tmp.rglob(f"{vid}.{kind}.m4a"), None)
            if f:
                data = f.read_bytes()
                requests.post(f"{kb.URL}/storage/v1/object/songs/{vid}.{kind}.m4a", data=data, timeout=300,
                              headers=kb.H({"Content-Type": "audio/mp4", "x-upsert": "true"})).raise_for_status()
                added += len(data)
        # count the stems in the song's size so the 900 MB limit of song_worker.py still holds
        row = kb.rest("GET", f"song_files?video_id=eq.{vid}&select=bytes", headers=kb.H())
        if row:
            kb.rest("PATCH", f"song_files?video_id=eq.{vid}", headers=kb.H({"Content-Type": "application/json"}),
                    json={"bytes": (row[0].get("bytes") or 0) + added})
        print(f"[ok] {res.get('seconds', {}).get(vid, '?')}s GPU · {vid}")
    for f in res.get("failed", []):
        print(f"[fallo] {f['id']}: {f['error']}")
    shutil.rmtree(tmp, ignore_errors=True)


def launch(user: str, jobs: list) -> str:
    batch = str(int(time.time()))
    code = (Path(__file__).with_name("kaggle") / "voz_kernel.py").read_text(encoding="utf-8")
    code = code.replace("JOBS = []", "JOBS = " + json.dumps(jobs), 1).replace('BATCH = ""', f'BATCH = "{batch}"', 1)
    code = code.replace("BUDGET_S = 6000", f"BUDGET_S = {kb.BUDGET_S}", 1)
    d = Path(tempfile.mkdtemp())
    (d / "voz_kernel.py").write_text(code, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG}", "title": SLUG, "code_file": "voz_kernel.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": [],
    }))
    kb.kaggle("kernels", "push", "-p", str(d), "-t", str(kb.BUDGET_S + 1800), check=True)
    return batch


def main():
    user = kb.load_creds()
    try:
        state = json.loads(kb.get_obj(STATE_OBJ) or b"{}")
        code, status = kb.kaggle("kernels", "status", f"{user}/{SLUG}")
        print("Kaggle voz:", status[-200:])
        low = status.lower()
        if code == 0 and ("running" in low or "queued" in low):
            return
        if state.get("batch") and not state.get("collected"):
            if "error" in low:
                print(kb.kaggle("kernels", "logs", f"{user}/{SLUG}")[1][-1500:])
            collect(user, state)
            state["collected"] = True
            kb.put_obj(STATE_OBJ, json.dumps(state).encode())
        jobs = next_jobs()
        if not jobs:
            print("todas las canciones tienen ya el beat separado")
            return
        try:
            batch = launch(user, jobs)
        except RuntimeError as e:
            print("no se pudo lanzar:", e)
            return
        kb.put_obj(STATE_OBJ, json.dumps({"batch": batch, "collected": False}).encode())
        print(f"lanzada tanda de voz {batch} con {len(jobs)} canciones")
    finally:
        kb.save_creds()


if __name__ == "__main__":
    main()
