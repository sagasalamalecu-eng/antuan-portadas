"""
Scene-aware motion for cover loops.

1. Understand the cover:
   - CLIP (zero-shot) → weather / atmosphere tags: rain, snow, fire, fog/smoke, night lights…
   - SegFormer (ADE20K) → region masks: sky, water, vegetation, animals, flags, people…
2. Animate each region in its own way, all with whole-number cycles per loop → seamless:
   - sky: slow large-scale drift          - water: small fast ripples
   - trees/grass/plants: wind sway        - animals: soft "breathing" motion
   - flags/curtains: waving               - lights: glow flicker
   - rain / snow: falling particle layers - fire: flicker + rising heat shimmer
   - fog/smoke: drifting haze
Runs on the GPU for the two models (a second or two) and numpy/OpenCV for the frames.
"""
import math

import cv2
import numpy as np
import torch
from PIL import Image

_models = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Region "wave" warping looked fake (text and objects rippling) → off until a real video model is used
WARP_REGIONS = False

# --------------------------------------------------------------------------- models

def _clip():
    if "clip" not in _models:
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
        _models["clip"] = (m, CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
    return _models["clip"]


def _seg():
    if "seg" not in _models:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        name = "nvidia/segformer-b2-finetuned-ade-512-512"
        m = SegformerForSemanticSegmentation.from_pretrained(name).to(DEVICE).eval()
        _models["seg"] = (m, SegformerImageProcessor.from_pretrained(name))
    return _models["seg"]


# (tag, prompt) — each tag is scored against a neutral baseline
TAGS = {
    "rain": "a photo with rain falling, raindrops and wet weather",
    "snow": "a photo with snow falling, snowflakes in the air",
    "fire": "a photo with fire and flames burning",
    "fog": "a photo with fog, mist or smoke in the air",
    "night_lights": "a night photo with glowing city lights and neon",
    "stars": "a night sky full of stars",
}
# "normal" descriptions absorb probability so a tag only wins when it really fits
NEUTRALS = [
    "a clear photo with calm weather",
    "a portrait photo of a person",
    "a graphic design album cover with text",
    "a sunny day outdoors",
    "an indoor photo",
    "a photo of a city street",
]

SEG_GROUPS = {
    "sky": ["sky"],
    "water": ["water", "sea", "river", "lake", "waterfall", "fountain", "swimming pool"],
    "plants": ["tree", "plant", "grass", "palm", "flower", "field"],
    "animal": ["animal"],
    "cloth": ["flag", "curtain", "awning", "canopy", "apparel"],
    "person": ["person"],
}


@torch.no_grad()
def analyze(img_rgb: np.ndarray) -> dict:
    pil = Image.fromarray(img_rgb)
    # --- CLIP tags
    clip, proc = _clip()
    prompts = list(TAGS.values()) + NEUTRALS
    inp = proc(text=prompts, images=pil, return_tensors="pt", padding=True).to(DEVICE)
    probs = clip(**inp).logits_per_image[0].softmax(0)  # all prompts compete
    tags = {k: float(probs[i]) for i, k in enumerate(TAGS)}

    # --- SegFormer masks
    seg, sproc = _seg()
    inp = sproc(images=pil, return_tensors="pt").to(DEVICE)
    out = seg(**inp).logits  # (1, 150, h/4, w/4)
    up = torch.nn.functional.interpolate(out, size=img_rgb.shape[:2], mode="bilinear", align_corners=False)
    labels = up.argmax(1)[0].cpu().numpy()
    id2label = seg.config.id2label
    masks = {}
    for group, names in SEG_GROUPS.items():
        ids = [i for i, n in id2label.items() if n.strip() in names]
        m = np.isin(labels, ids).astype(np.float32)
        if m.mean() > 0.01:  # ignore tiny noisy bits
            masks[group] = m
    return {"tags": tags, "masks": masks}


# --------------------------------------------------------------------------- helpers

def _feather(mask: np.ndarray, px: int) -> np.ndarray:
    k = max(3, px | 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


def _noise(h, w, scale, seed):
    """Smooth random field in [-1, 1] with feature size ≈ scale px."""
    rng = np.random.default_rng(seed)
    sh, sw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    small = rng.standard_normal((sh, sw)).astype(np.float32)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    big /= (np.abs(big).max() + 1e-6)
    return big


class Field:
    """Periodic displacement: d(t) = A·[n1·cos(2πkt) + n2·sin(2πkt)] (+ optional constant direction wobble)."""

    def __init__(self, h, w, scale, amp, cycles, seed, weight, vertical=1.0):
        self.nx1, self.nx2 = _noise(h, w, scale, seed), _noise(h, w, scale, seed + 1)
        self.ny1, self.ny2 = _noise(h, w, scale, seed + 2), _noise(h, w, scale, seed + 3)
        self.amp, self.k, self.w, self.v = amp, cycles, weight, vertical

    def at(self, ph):
        c, s = math.cos(2 * math.pi * self.k * ph), math.sin(2 * math.pi * self.k * ph)
        dx = (self.nx1 * c + self.nx2 * s) * self.amp * self.w
        dy = (self.ny1 * c + self.ny2 * s) * self.amp * self.w * self.v
        return dx, dy


def _particles(h, w, n, seed, size, speed_cycles, angle_deg, alpha, kind):
    """Pre-generated particle set; position at phase ph wraps exactly → seamless loop."""
    rng = np.random.default_rng(seed)
    return dict(
        x=rng.uniform(0, w, n), y=rng.uniform(0, h, n), z=rng.uniform(0.5, 1.0, n),
        ph=rng.uniform(0, 1, n), h=h, w=w, size=size, k=speed_cycles,
        slope=math.tan(math.radians(angle_deg)), alpha=alpha, kind=kind,
    )


def _draw_particles(P, ph):
    h, w = P["h"], P["w"]
    layer = np.zeros((h, w), np.float32)
    dy = P["k"] * h * ph  # integer number of screen heights per loop
    for x0, y0, z, p0 in zip(P["x"], P["y"], P["z"], P["ph"]):
        y = (y0 + dy * z if P["kind"] == "snow" else y0 + dy) % h
        if P["kind"] == "rain":
            x = (x0 + P["slope"] * (y - y0)) % w
            L = P["size"] * (0.6 + 0.8 * z)
            x2, y2 = x + P["slope"] * L, y + L
            cv2.line(layer, (int(x), int(y)), (int(x2), int(y2)), float(P["alpha"] * z), 1, cv2.LINE_AA)
        else:  # snow: falls (speed by depth) and sways sideways
            x = (x0 + 10 * z * math.sin(2 * math.pi * (2 * ph + p0))) % w
            cv2.circle(layer, (int(x), int(y)), max(1, int(P["size"] * z)), float(P["alpha"] * z), -1, cv2.LINE_AA)
    return layer


# --------------------------------------------------------------------------- main

def apply(frames: list, img_rgb: np.ndarray, info: dict | None = None) -> list:
    """frames: list of HxWx3 uint8 (already head-animated or just copies of the cover)."""
    info = info or analyze(img_rgb)
    tags, masks = info["tags"], info["masks"]
    h, w = img_rgb.shape[:2]
    n = len(frames)
    S = max(h, w) / 512.0  # effects are tuned for 512 px

    # ---- displacement fields per region
    fields = []
    if not WARP_REGIONS:
        masks = {}
    if "sky" in masks:
        fields.append(Field(h, w, 180 * S, 5.0 * S, 1, 11, _feather(masks["sky"], int(31 * S)), vertical=0.35))
    if "water" in masks:
        fields.append(Field(h, w, 22 * S, 2.2 * S, 3, 21, _feather(masks["water"], int(15 * S)), vertical=0.6))
    if "plants" in masks:
        fields.append(Field(h, w, 60 * S, 2.6 * S, 2, 31, _feather(masks["plants"], int(21 * S)), vertical=0.3))
    if "animal" in masks:
        fields.append(Field(h, w, 120 * S, 1.6 * S, 1, 41, _feather(masks["animal"], int(25 * S)), vertical=0.8))
    if "cloth" in masks:
        fields.append(Field(h, w, 45 * S, 2.4 * S, 2, 51, _feather(masks["cloth"], int(15 * S)), vertical=0.5))
    fire = tags.get("fire", 0) > 0.3
    fire_mask = None
    if fire:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        fm = ((hsv[..., 0] < 25) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 150)
        fire_mask = _feather(fm.astype(np.float32), int(21 * S))
        fields.append(Field(h, w, 25 * S, 2.5 * S, 3, 61, fire_mask, vertical=1.5))

    # ---- light spots (much brighter than surroundings)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    local = cv2.GaussianBlur(gray, (0, 0), 12 * S)
    spots = np.clip((gray - local - 0.12) * 4, 0, 1) * (gray > 0.6)
    spots = _feather(spots.astype(np.float32), int(9 * S))
    spot_phase = (_noise(h, w, 40 * S, 71) + 1) / 2
    lights_on = spots.mean() > 0.0005

    # ---- weather layers
    parts = []
    if tags.get("rain", 0) > 0.3:
        parts.append((_particles(h, w, int(700 * S * S), 81, 22 * S, 3, -8, 0.35, "rain"), (0.82, 0.86, 0.95)))
        parts.append((_particles(h, w, int(400 * S * S), 82, 12 * S, 2, -8, 0.18, "rain"), (0.75, 0.8, 0.9)))
    if tags.get("snow", 0) > 0.3:
        parts.append((_particles(h, w, int(260 * S * S), 91, 2.2 * S, 1, 0, 0.85, "snow"), (1, 1, 1)))
    fog = tags.get("fog", 0) > 0.35
    if fog:
        fog_a, fog_b = (_noise(h, w, 140 * S, 101) + 1) / 2, (_noise(h, w, 140 * S, 102) + 1) / 2

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2, h / 2
    out = []
    for i, f in enumerate(frames):
        ph = i / n
        # gentle global "breath" zoom keeps the whole cover alive
        z = 1.0 + 0.012 * (0.5 - 0.5 * math.cos(2 * math.pi * ph))
        mx = cx + (xx - cx) / z
        my = cy + (yy - cy) / z
        for fd in fields:
            dx, dy = fd.at(ph)
            mx = mx + dx
            my = my + dy
        frame = cv2.remap(f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255

        if lights_on:
            flick = 0.5 + 0.5 * np.sin(2 * np.pi * (2 * ph + spot_phase * 3))
            frame *= (1 + spots * (flick - 0.4) * 0.6)[..., None]
        if fire_mask is not None:
            fl = 1 + 0.12 * math.sin(2 * math.pi * 5 * ph) + 0.06 * math.sin(2 * math.pi * 11 * ph + 1)
            frame *= (1 + fire_mask * (fl - 1))[..., None]
        if fog:
            c, s = math.cos(2 * math.pi * ph), math.sin(2 * math.pi * ph)
            haze = np.clip(fog_a * (0.5 + 0.5 * c) + fog_b * (0.5 + 0.5 * s), 0, 1) * 0.12
            frame = frame * (1 - haze[..., None]) + haze[..., None] * 0.9
        for P, col in parts:
            layer = _draw_particles(P, ph)
            if P["kind"] == "rain":
                layer = cv2.GaussianBlur(layer, (0, 0), 0.6)
            frame = 1 - (1 - frame) * (1 - layer[..., None] * np.array(col, np.float32))  # screen blend

        out.append((np.clip(frame, 0, 1) * 255).astype(np.uint8))
    return out


def describe(info: dict) -> str:
    t = ", ".join(f"{k} {v:.2f}" for k, v in info["tags"].items() if v > 0.1)
    m = ", ".join(f"{k} {v.mean() * 100:.0f}%" for k, v in info["masks"].items())
    return f"tags[{t or '-'}] zones[{m or '-'}]"
