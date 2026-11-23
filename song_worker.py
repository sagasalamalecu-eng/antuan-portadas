"""
Antuan Music · canciones para escuchar con la pantalla apagada (y grabar vídeo con audio limpio).

El móvil deja encargos en la tabla song_files (supabase/songs.sql). Este programa baja el audio con yt-dlp,
lo sube al almacén PRIVADO "songs" de Supabase y borra la copia local. Cuando el almacén pasa del límite,
borra las que hace más que no se escuchan.

Dónde corre: en GitHub Actions (repo antuan-portadas, .github/workflows/canciones.yml, `--drain`): Supabase
avisa al pedir una canción, procesa la cola y se apaga, sin el PC. Respaldo en el PC: doble clic en D:\\AntuanMusic\\arrancar_canciones.bat
Config: D:\\AntuanAI\\.env (SUPABASE_URL y SUPABASE_SERVICE_KEY, las mismas que la IA de portadas).
Necesita yt-dlp (y deno/ffmpeg) instalados: winget install yt-dlp.yt-dlp
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import requests

ENV = Path(os.environ.get("ANTUAN_ENV", "D:/AntuanAI/.env"))
POLL_SECONDS = 4
LIMIT_MB = float(os.environ.get("SONGS_LIMIT_MB", "900"))  # plan gratis de Supabase: 1 GB en total
# m4a when there is one; otherwise any audio, or even a video with sound (ffmpeg then extracts the audio)
FORMAT = "bestaudio[ext=m4a]/bestaudio[acodec^=mp4a]/bestaudio/best"


def load_env(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


load_env(ENV)
URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def H(extra=None):
    # new secret keys (sb_secret_...) only in "apikey"; legacy service_role JWTs also as Bearer
    h = {"apikey": KEY}
    if not KEY.startswith("sb_"):
        h["Authorization"] = f"Bearer {KEY}"
    h.update(extra or {})
    return h


def rest(method, path, **kw):
    r = requests.request(method, f"{URL}/rest/v1/{path}", timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def claim():
    rows = rest("GET", "song_files?status=eq.queued&order=requested_at.asc&limit=1&select=video_id", headers=H())
    if not rows:
        return None
    vid = rows[0]["video_id"]
    got = rest("PATCH", f"song_files?video_id=eq.{vid}&status=eq.queued",
               headers=H({"Prefer": "return=representation", "Content-Type": "application/json"}),
               json={"status": "processing", "updated_at": "now()"})
    return got[0]["video_id"] if got else None


def finish(vid, **fields):
    fields["updated_at"] = "now()"
    rest("PATCH", f"song_files?video_id=eq.{vid}", headers=H({"Content-Type": "application/json"}), json=fields)


def requeue_stale():
    rest("PATCH", "song_files?status=eq.processing", headers=H({"Content-Type": "application/json"}),
         json={"status": "queued"})


def ytdlp() -> str:
    exe = shutil.which("yt-dlp")
    if not exe:
        sys.exit("No encuentro yt-dlp. Instálalo con: winget install yt-dlp.yt-dlp")
    return exe


def download(vid: str, folder: Path) -> Path:
    cmd = [ytdlp(), "-q", "--no-playlist", "--no-progress", "-f", FORMAT,
           "-x", "--audio-format", "m4a", "--max-filesize", "24M",
           "-o", str(folder / "%(id)s.%(ext)s")]
    # servers (GitHub) get "confirm you're not a bot": they go signed in with a SEPARATE Google account's cookies
    cookies = os.environ.get("YT_COOKIES_FILE", "")
    if cookies and Path(cookies).exists() and Path(cookies).stat().st_size > 0:
        cmd += ["--cookies", cookies]
    cmd.append(f"https://www.youtube.com/watch?v={vid}")
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    files = sorted(folder.glob(f"{vid}.*"), key=lambda f: f.stat().st_size, reverse=True)
    m4a = [f for f in files if f.suffix == ".m4a"]
    if p.returncode != 0 or not m4a:
        out = (p.stderr or p.stdout or "yt-dlp falló").strip()
        if "cookies are no longer valid" in out:
            # say it plainly in the table: the fix is exporting the cookies again (see ESTADO.md)
            raise RuntimeError("COOKIES CADUCADAS: exporta otra vez las cookies de la cuenta aparte y guárdalas en YT_COOKIES")
        print(out[-1500:])
        lines = [l for l in out.splitlines() if l.strip()]
        raise RuntimeError(" | ".join(lines[-3:])[:480])
    return m4a[0]


def upload(vid: str, file: Path) -> str:
    path = f"{vid}.m4a"
    with open(file, "rb") as fh:
        r = requests.post(f"{URL}/storage/v1/object/songs/{path}", data=fh, timeout=300,
                          headers=H({"Content-Type": "audio/mp4", "x-upsert": "true", "cache-control": "31536000"}))
    r.raise_for_status()
    return path


def evict():
    """Over the limit → delete the songs nobody has played for the longest time."""
    rows = rest("GET", "song_files?status=eq.done&order=last_played.asc&select=video_id,path,bytes", headers=H()) or []
    total = sum(r.get("bytes") or 0 for r in rows)
    limit = LIMIT_MB * 1024 * 1024
    for r in rows:
        if total <= limit:
            break
        try:
            requests.delete(f"{URL}/storage/v1/object/songs/{r['path']}", headers=H(), timeout=30)
            rest("DELETE", f"song_files?video_id=eq.{r['video_id']}", headers=H())
            total -= r.get("bytes") or 0
            print(f"   [espacio] borrada {r['video_id']} (la que menos se escucha)")
        except Exception as e:
            print("   [espacio] no se pudo borrar:", e)


def work(vid: str):
    t = time.time()
    folder = Path(tempfile.mkdtemp(prefix="antuan-song-"))
    try:
        f = download(vid, folder)
        size = f.stat().st_size
        path = upload(vid, f)
        finish(vid, status="done", path=path, bytes=size, error=None)
        print(f"[ok] {vid}  {size / 1048576:.1f} MB  {time.time() - t:.0f}s")
        evict()
    except Exception as e:
        print(f"[fallo] {vid}: {e}")
        finish(vid, status="failed", error=str(e)[:500])
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def drain(max_seconds: float = 20 * 60) -> int:
    """Process the queue until it's empty or time runs out, then exit (cloud server)."""
    if not URL or not KEY:
        sys.exit("Falta SUPABASE_URL o SUPABASE_SERVICE_KEY")
    ytdlp()
    requeue_stale()
    start, n = time.time(), 0
    while time.time() - start < max_seconds:
        vid = claim()
        if not vid:
            # a request may arrive while we finish: look once more before shutting down
            time.sleep(5)
            vid = claim()
            if not vid:
                break
        work(vid)
        n += 1
    return n


def run():
    if not URL or not KEY:
        sys.exit(f"Falta SUPABASE_URL o SUPABASE_SERVICE_KEY en {ENV}")
    ytdlp()
    try:
        requeue_stale()
    except requests.HTTPError as e:
        sys.exit(f"Supabase no tiene la tabla song_files: ejecuta supabase/songs.sql en el SQL Editor ({e})")
    print("[canciones] esperando encargos del móvil... (deja esta ventana abierta)")
    while True:
        try:
            vid = claim()
        except Exception as e:
            print("[canciones] Supabase no responde:", e)
            time.sleep(15)
            continue
        if not vid:
            time.sleep(POLL_SECONDS)
            continue
        try:
            work(vid)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    if "--drain" in sys.argv:
        print("procesadas:", drain())
    else:
        run()
