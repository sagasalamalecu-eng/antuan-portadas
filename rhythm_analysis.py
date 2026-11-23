"""
Antuan Music · Modo Teclas: análisis de ritmo de una canción → songs/{videoId}.ritmo.json (almacén privado).

Saca, sin GPU y en ~20-40 s por canción:
  - beats: tiempos del compás (librosa beat_track)
  - onsets: golpes [t, fuerza 0-1] (bombo, caja, ataques de voz)
  - notes: melodía aproximada de la voz [inicio, duración, nota MIDI] (pYIN sobre la parte armónica, 80-1000 Hz)
La app (src/features/game/chart.ts) lo usa para que las teclas caigan con el ritmo real y el carril siga la melodía.
Uso suelto: python rhythm_analysis.py cancion.m4a > ritmo.json
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SR = 22050


def _wav(src: Path, dst: Path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ac", "1", "-ar", str(SR), str(dst)],
                   check=True, timeout=180)


def analyze(audio: Path) -> dict:
    import numpy as np
    import librosa

    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        _wav(audio, wav)
        y, sr = librosa.load(str(wav), sr=SR, mono=True)
    dur = float(len(y) / sr)
    hop = 512

    harm, perc = librosa.effects.hpss(y)
    env = librosa.onset.onset_strength(y=perc, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, units="time")
    on_frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, backtrack=False)
    strength = env[on_frames] if len(on_frames) else np.array([])
    top = float(np.percentile(strength, 95)) if len(strength) else 1.0
    onsets = [[round(float(t), 3), round(min(1.0, float(s) / max(top, 1e-6)), 2)]
              for t, s in zip(librosa.frames_to_time(on_frames, sr=sr, hop_length=hop), strength)]

    # vocal-ish melody: pYIN on the harmonic part, grouped into notes (same semitone, voiced)
    notes = []
    try:
        f0, voiced, prob = librosa.pyin(harm, fmin=80, fmax=1000, sr=sr, frame_length=2048, hop_length=hop)
        times = librosa.times_like(f0, sr=sr, hop_length=hop)
        midi = np.where(voiced & (prob > 0.35), librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
        cur = None
        for t, m in zip(times, midi):
            m = None if np.isnan(m) else int(round(float(m)))
            if cur and m is not None and abs(m - cur[2]) <= 1:
                cur[1] = t - cur[0]
                continue
            if cur and cur[1] >= 0.08:
                notes.append([round(float(cur[0]), 3), round(float(cur[1]), 3), int(cur[2])])
            cur = [t, 0.0, m] if m is not None else None
        if cur and cur[1] >= 0.08:
            notes.append([round(float(cur[0]), 3), round(float(cur[1]), 3), int(cur[2])])
    except Exception as e:  # melody is optional
        print("   [ritmo] sin melodía:", e)

    bpm = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    return {
        "v": 1,
        "dur": round(dur, 2),
        "bpm": round(bpm, 1),
        "beats": [round(float(b), 3) for b in beats],
        "onsets": onsets,
        "notes": notes,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1]))))
