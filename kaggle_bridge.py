"""
Antuan Music · puente GitHub Actions ⇄ Kaggle (GPU gratis, 30 h/semana, sin tarjeta).

Cada vez que corre (workflow "Kaggle Wan", cada 2 h):
 1. Si el cuaderno de Kaggle sigue trabajando → no hace nada.
 2. Si terminó → baja sus vídeos y los sube a Supabase (loops/w5-…mp4), para siempre.
    Las IA antiguas de Hugging Face ("ai-") se guardan en loops/backup/; los bucles antiguos sin IA se borran.
 3. Prepara la siguiente tanda (portadas que aún no tienen vídeo Wan, las más recientes primero) y la lanza.
Si se gasta el cupo semanal de Kaggle, el lanzamiento falla y se reintenta en la siguiente pasada:
sigue solo cuando Kaggle reinicia el cupo.

La sesión de Kaggle (credentials.json) vive en el almacén privado "worker" de Supabase porque Kaggle la renueva;
la primera vez sale del secreto KAGGLE_CREDENTIALS.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

URL = os.environ["SUPABASE_URL"].rstrip("/")
KEY = os.environ["SUPABASE_SERVICE_KEY"]
CRED_OBJ = "worker/kaggle_credentials.json"
STATE_OBJ = "worker/kaggle_state.json"
CRED = Path.home() / ".kaggle" / "credentials.json"
SLUG = "antuan-wan"
BATCH_SIZE = int(os.environ.get("KAGGLE_BATCH", "30"))
BUDGET_S = int(os.environ.get("KAGGLE_BUDGET_S", str(int(1.8 * 3600))))


def H(extra=None):
    h = {"apikey": KEY}
    if not KEY.startswith("sb_"):
        h["Authorization"] = f"Bearer {KEY}"
    h.update(extra or {})
    return h


def rest(method, path, **kw):
    r = requests.request(method, f"{URL}/rest/v1/{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def get_obj(path):
    r = requests.get(f"{URL}/storage/v1/object/{path}", headers=H(), timeout=30)
    return r.content if r.status_code == 200 else None


def put_obj(path, data: bytes, ctype="application/json"):
    requests.post(f"{URL}/storage/v1/object/{path}", data=data, timeout=60,
                  headers=H({"Content-Type": ctype, "x-upsert": "true"})).raise_for_status()


def kaggle(*args, check=False):
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    if check and p.returncode:
        raise RuntimeError(out[-800:])
    return p.returncode, out


def load_creds() -> str:
    CRED.parent.mkdir(exist_ok=True)
    data = get_obj(CRED_OBJ) or os.environ.get("KAGGLE_CREDENTIALS", "").encode()
    if not data:
        sys.exit("Sin sesión de Kaggle (secreto KAGGLE_CREDENTIALS).")
    CRED.write_bytes(data)
    os.chmod(CRED, 0o600)
    return json.loads(data)["username"]


def save_creds():
    if CRED.exists():
        put_obj(CRED_OBJ, CRED.read_bytes())


def collect(user: str, state: dict):
    """Download the finished batch and publish its videos."""
    tmp = Path(tempfile.mkdtemp())
    code, out = kaggle("kernels", "output", f"{user}/{SLUG}", "-p", str(tmp), "-o")
    res_file = next(tmp.rglob("results.json"), None)
    if not res_file:
        print("sin results.json en la salida:", out[-400:])
        return
    res = json.loads(res_file.read_text())
    if res.get("batch") != state.get("batch"):
        print("la salida es de otra tanda, no se sube")
        return
    old = {j["key"]: j.get("old") for j in state.get("jobs", [])}
    ok = 0
    for d in res.get("done", []):
        key = d["key"]
        mp4 = next(tmp.rglob(f"{key}.mp4"), None)
        if not mp4:
            continue
        row = rest("GET", f"cover_loops?cover_key=eq.{key}&select=video_path", headers=H())
        cur = row[0]["video_path"] if row else None
        if cur and cur.startswith("w5-"):
            continue  # already upgraded meanwhile
        path = f"w5-{key}-{int(time.time())}.mp4"
        requests.post(f"{URL}/storage/v1/object/loops/{path}", data=mp4.read_bytes(), timeout=120,
                      headers=H({"Content-Type": "video/mp4", "x-upsert": "true", "cache-control": "31536000"})
                      ).raise_for_status()
        rest("PATCH", f"cover_loops?cover_key=eq.{key}", headers=H({"Content-Type": "application/json"}),
             json={"status": "done", "video_path": path, "error": None, "updated_at": "now()"})
        prev = cur or old.get(key)
        if prev and prev != path:
            if prev.startswith("ai-"):
                requests.post(f"{URL}/storage/v1/object/move", timeout=30, headers=H({"Content-Type": "application/json"}),
                              json={"bucketId": "loops", "sourceKey": prev, "destinationKey": f"backup/{prev}"})
            else:
                requests.delete(f"{URL}/storage/v1/object/loops/{prev}", headers=H(), timeout=30)
        ok += 1
        print(f"[ok] {res.get('seconds', {}).get(key, '?')}s GPU · {key} · {d.get('caption', '')}")
    for f in res.get("failed", []):
        print(f"[fallo] {f['key']}: {f['error']}")
    secs = list(res.get("seconds", {}).values())
    if secs:
        print(f"subidas {ok} · media {sum(secs) / len(secs):.0f}s por portada en la GPU de Kaggle")
    shutil.rmtree(tmp, ignore_errors=True)


def next_jobs() -> list:
    rows = rest("GET", "cover_loops?status=eq.done&video_path=not.like.w5-*&order=requested_at.desc"
                       f"&limit={BATCH_SIZE}&select=cover_key,cover_url,video_path", headers=H()) or []
    return [{"key": r["cover_key"], "url": r["cover_url"], "old": r.get("video_path")} for r in rows]


def launch(user: str, jobs: list) -> str:
    batch = str(int(time.time()))
    src = Path(__file__).with_name("kaggle") / "wan_kernel.py"
    code = src.read_text(encoding="utf-8")
    code = code.replace("JOBS = []", "JOBS = " + json.dumps([{"key": j["key"], "url": j["url"]} for j in jobs]), 1)
    code = code.replace('BATCH = ""', f'BATCH = "{batch}"', 1)
    code = code.replace("BUDGET_S = 6000", f"BUDGET_S = {BUDGET_S}", 1)
    d = Path(tempfile.mkdtemp())
    (d / "wan_kernel.py").write_text(code, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG}", "title": SLUG, "code_file": "wan_kernel.py", "language": "python",
        "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "true", "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "model_sources": [],
    }))
    kaggle("kernels", "push", "-p", str(d), "--accelerator", "NvidiaTeslaT4", "-t", str(BUDGET_S + 1800), check=True)
    return batch


def main():
    user = load_creds()
    try:
        state = json.loads(get_obj(STATE_OBJ) or b"{}")
        code, status = kaggle("kernels", "status", f"{user}/{SLUG}")
        print("Kaggle:", status[-300:])
        low = status.lower()
        if code == 0 and ("running" in low or "queued" in low):
            print("sigue trabajando; nada que hacer")
            return
        if state.get("batch") and not state.get("collected"):
            if "error" in low or "cancel" in low:
                _, log = kaggle("kernels", "logs", f"{user}/{SLUG}")
                print("la tanda anterior falló:", log[-1500:])
            collect(user, state)  # also on error: whatever finished before the error is published
            state["collected"] = True
            put_obj(STATE_OBJ, json.dumps(state).encode())
        _, quota = kaggle("quota")
        print(quota)
        jobs = next_jobs()
        if not jobs:
            print("todas las portadas tienen ya vídeo Wan")
            return
        try:
            batch = launch(user, jobs)
        except RuntimeError as e:
            print("no se pudo lanzar (¿cupo semanal gastado?):", e)
            return
        put_obj(STATE_OBJ, json.dumps({"batch": batch, "jobs": jobs, "collected": False}).encode())
        print(f"lanzada tanda {batch} con {len(jobs)} portadas")
    finally:
        save_creds()


if __name__ == "__main__":
    main()
