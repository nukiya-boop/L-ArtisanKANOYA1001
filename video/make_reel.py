"""L'Artisan KANOYA — Instagram リール用 PR 動画（1080x1920 / 30fps / 30秒）

ある日のディナーより「黄人参、たいら貝、シナモンリーフ」

使い方:
    python3 video/make_reel.py --fonts <フォントディレクトリ> --music <BGM.mp3> --out video/reel.mp4

必要: Pillow, numpy, imageio-ffmpeg
フォント（Google Fonts / OFL）: Shippori Mincho (Regular, Medium), Cormorant Garamond (Regular, Italic)
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import unicodedata
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 30
DURATION = 30.0
NFRAMES = int(FPS * DURATION)
PRELOAD_SCALE = 2 / 3  # 5760x3840 -> 3840x2560

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "images")

GOLD = (222, 204, 164)
IVORY = (244, 239, 228)

# ---------------------------------------------------------------- timeline
# ショットの境界（秒）。各境界の前後 TR 秒でトランジション。
BOUNDS = [0.0, 4.4, 8.0, 11.6, 15.2, 18.8, 22.0, 25.2, 30.0]
TR = 0.5

# カメラ: 中心 (x, y) は画像に対する割合、h は切り出す高さの割合
SHOTS = [
    dict(img="0180", c0=(0.50, 0.47), c1=(0.50, 0.46), h0=1.00, h1=0.80),  # 引きの全景
    dict(img="0187", c0=(0.50, 0.40), c1=(0.52, 0.56), h0=0.92, h1=0.84),  # 縦位置・寄り
    dict(img="0181", c0=(0.36, 0.52), c1=(0.60, 0.55), h0=0.92, h1=0.82),  # 黄人参
    dict(img="0196", c0=(0.74, 0.62), c1=(0.66, 0.58), h0=0.60, h1=0.74),  # たいら貝
    dict(img="0188", c0=(0.60, 0.30), c1=(0.56, 0.48), h0=0.78, h1=0.92),  # シナモンリーフ
    dict(img="0166", c0=(0.38, 0.50), c1=(0.58, 0.50), h0=1.00, h1=0.96),  # 余白・情景
    dict(img="0178", c0=(0.52, 0.52), c1=(0.52, 0.50), h0=0.70, h1=0.92),  # 引いていく
    dict(img="0192", c0=(0.52, 0.52), c1=(0.50, 0.50), h0=0.74, h1=1.00),  # 真上・エンド
]
# 各境界のトランジション
TRANSITIONS = ["blur", "light_wipe", "zoom", "sudare", "blur", "light_wipe_r", "bloom"]


def ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(u, 0.0), 1.0))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- images
_IMAGES = {}


def load_images():
    for path in glob.glob(os.path.join(IMG_DIR, "*.jpg")):
        name = unicodedata.normalize("NFC", os.path.basename(path))
        key = "".join(ch for ch in name if ch.isdigit())[-4:]
        im = Image.open(path).convert("RGB")
        im = im.resize((round(im.width * PRELOAD_SCALE), round(im.height * PRELOAD_SCALE)), Image.LANCZOS)
        _IMAGES[key] = im
    missing = {s["img"] for s in SHOTS} - set(_IMAGES)
    if missing:
        raise SystemExit(f"images not found: {missing}")


def render_shot(i, t, extra_zoom=1.0):
    s = SHOTS[i]
    im = _IMAGES[s["img"]]
    t0, t1 = BOUNDS[i] - TR, BOUNDS[i + 1] + TR
    u = (t - t0) / (t1 - t0)
    u = 0.65 * min(max(u, 0), 1) + 0.35 * ease(u)
    cx = (s["c0"][0] + (s["c1"][0] - s["c0"][0]) * u) * im.width
    cy = (s["c0"][1] + (s["c1"][1] - s["c0"][1]) * u) * im.height
    h = (s["h0"] + (s["h1"] - s["h0"]) * u) * im.height / extra_zoom
    h = min(h, im.height, im.width * 16 / 9)
    w = h * 9 / 16
    x0 = min(max(cx - w / 2, 0), im.width - w)
    y0 = min(max(cy - h / 2, 0), im.height - h)
    frame = im.resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h))
    return np.asarray(frame, dtype=np.float32)


# ---------------------------------------------------------------- transitions
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)


def blur(a, r):
    if r < 0.5:
        return a
    small = Image.fromarray(a.astype(np.uint8)).resize((W // 2, H // 2), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(r / 2))
    return np.asarray(small.resize((W, H), Image.BILINEAR), dtype=np.float32)


def transition(kind, i, t, p):
    """shot i -> i+1 の合成。p は 0..1。"""
    if kind == "zoom":
        a = render_shot(i, t, extra_zoom=1 + 0.18 * p)
        b = render_shot(i + 1, t, extra_zoom=1.12 - 0.12 * p)
        k = ease(p)
        return blur(a, 14 * k) * (1 - k) + blur(b, 14 * (1 - k)) * k

    a = render_shot(i, t)
    b = render_shot(i + 1, t)
    if kind == "blur":
        k = ease(p)
        r = 26 * math.sin(math.pi * p)
        return blur(a, r) * (1 - k) + blur(b, r) * k
    if kind in ("light_wipe", "light_wipe_r"):
        # 斜めに光が走り抜けて次のカットへ
        d = (_XX * 0.55 + _YY * 0.35) / (W * 0.55 + H * 0.35)
        if kind == "light_wipe_r":
            d = 1 - (_XX * 0.55 + (H - _YY) * 0.35) / (W * 0.55 + H * 0.35)
        pos = -0.25 + 1.5 * ease(p)
        m = smoothstep(pos + 0.12, pos - 0.12, d)[..., None]
        glow = np.exp(-((d - pos) / 0.07) ** 2)[..., None] * math.sin(math.pi * p)
        out = a * (1 - m) + b * m
        return out + glow * np.array([255, 225, 170], np.float32) * 0.55
    if kind == "sudare":
        # 簾（すだれ）が上から順に下りるように、縦の短冊で切り替え
        n = 6
        col = np.floor(_XX / (W / n))
        local = np.clip(p * 1.7 - col * 0.14, 0, 1)
        edge = local * (H + 240) - 120
        m = smoothstep(edge + 120, edge - 120, _YY)[..., None]
        return a * (1 - m) + b * m
    if kind == "bloom":
        k = ease(p)
        out = a * (1 - k) + b * k
        g = math.sin(math.pi * p)
        return out + blur(out, 30) * 0.35 * g
    raise ValueError(kind)


# ---------------------------------------------------------------- typography
class Fonts:
    dir = None

    @staticmethod
    @lru_cache(None)
    def get(name, size):
        files = {
            "jp": "Shippori-Medium.ttf",
            "jp_light": "Shippori-Regular.ttf",
            "latin": "Cormorant-Regular.ttf",
            "italic": "Cormorant-Italic.ttf",
        }
        return ImageFont.truetype(os.path.join(Fonts.dir, files[name]), size)


@lru_cache(None)
def glyph_sprite(ch, font, size, vertical, color):
    """1文字分のスプライト（影つき）を作る。(RGBA配列, 原点オフセット)"""
    f = Fonts.get(font, size)
    pad = size
    canvas = Image.new("L", (size * 3, size * 3))
    d = ImageDraw.Draw(canvas)
    if vertical:
        d.text((pad + size / 2, pad), ch, font=f, fill=255, direction="ttb", features=["vert"], anchor="mt")
    else:
        d.text((pad, pad + size), ch, font=f, fill=255, anchor="ls")
    alpha = np.asarray(canvas, np.float32) / 255
    shadow = np.asarray(canvas.filter(ImageFilter.GaussianBlur(max(size * 0.22, 6))), np.float32) / 255
    shadow = np.clip(shadow * 1.8, 0, 1)
    rgba = np.zeros(alpha.shape + (4,), np.float32)
    # 影を下に、文字を上に合成済みの状態で保持
    sa = shadow * 0.7
    out_a = alpha + sa * (1 - alpha)
    rgb = (np.array(color, np.float32) * alpha[..., None]) / np.maximum(out_a[..., None], 1e-6)
    rgba[..., :3] = rgb
    rgba[..., 3] = out_a
    return rgba, pad


def layout(text, font, size, x, y, vertical=False, tracking=0.0, align="center", color=IVORY):
    """文字ごとの配置を返す。vertical のとき (x, y) は列の中心・上端。"""
    f = Fonts.get(font, size)
    glyphs = []
    if vertical:
        step = size * (1.0 + tracking)
        for k, ch in enumerate(text):
            glyphs.append((ch, x - size / 2, y + k * step))
    else:
        adv = [f.getlength(ch) + size * tracking for ch in text]
        total = sum(adv) - size * tracking
        cx = {"center": x - total / 2, "left": x, "right": x - total}[align]
        for ch, a in zip(text, adv):
            glyphs.append((ch, cx, y))
            cx += a
    return dict(glyphs=glyphs, font=font, size=size, vertical=vertical, color=color)


class Caption:
    def __init__(self, t_in, t_out, items, stagger=0.055, fade=0.7, drift=18, order_offset=0):
        self.t_in, self.t_out = t_in, t_out
        self.items = items
        self.stagger, self.fade, self.drift = stagger, fade, drift
        self.order_offset = order_offset


def draw_captions(frame, t, captions, lines):
    for cap in captions:
        if not (cap.t_in <= t <= cap.t_out + 0.1):
            continue
        out_k = 1 - ease((t - (cap.t_out - 0.6)) / 0.6)
        n = 0
        for item in cap.items:
            for ch, gx, gy in item["glyphs"]:
                local = (t - cap.t_in - n * cap.stagger) / cap.fade
                n += 1
                if local <= 0 or ch == " ":
                    continue
                k = ease(local) * out_k
                if k <= 0.003:
                    continue
                sprite, pad = glyph_sprite(ch, item["font"], item["size"], item["vertical"], item["color"])
                dy = (1 - ease(local)) * cap.drift
                px = int(round(gx - pad))
                py = int(round(gy - pad + dy - (item["size"] if not item["vertical"] else 0)))
                paste(frame, sprite, px, py, k)
    for (t_in, t_out, x, y, length, vertical) in lines:
        if not (t_in <= t <= t_out):
            continue
        grow = ease((t - t_in) / 0.9)
        k = 1 - ease((t - (t_out - 0.6)) / 0.6)
        L = length * grow
        if vertical:
            y0, y1, x0, x1 = int(y), int(y + L), int(x), int(x) + 2
        else:
            x0, x1, y0, y1 = int(x - L / 2), int(x + L / 2), int(y), int(y) + 2
        if x1 > x0 and y1 > y0:
            region = frame[y0:y1, x0:x1]
            region[:] = region * (1 - 0.85 * k) + np.array(GOLD, np.float32) * 0.85 * k


def paste(frame, sprite, px, py, k):
    sh, sw = sprite.shape[:2]
    x0, y0 = max(px, 0), max(py, 0)
    x1, y1 = min(px + sw, W), min(py + sh, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = sprite[y0 - py:y1 - py, x0 - px:x1 - px]
    a = s[..., 3:4] * k
    region = frame[y0:y1, x0:x1]
    region[:] = region * (1 - a) + s[..., :3] * a


def build_captions():
    caps, lines = [], []
    B = BOUNDS

    # 1. オープニング
    caps.append(Caption(0.9, B[1] - 0.2, [
        layout("L'Artisan KANOYA", "latin", 76, W / 2, 1500, tracking=0.08, color=IVORY),
    ], stagger=0.045, fade=0.9))
    caps.append(Caption(1.8, B[1] - 0.2, [
        layout("ある日のディナーより", "jp_light", 40, W / 2, 1610, tracking=0.35, color=GOLD),
    ], stagger=0.06, fade=0.8, drift=10))
    lines.append((1.4, B[1] - 0.2, W / 2, 1535, 220, False))

    # 2. 情景コピー（縦書き）
    caps.append(Caption(B[1] + 0.3, B[2] - 0.1, [
        layout("闇に灯る、", "jp", 58, 930, 240, vertical=True, tracking=0.18),
        layout("季節の彩り。", "jp", 58, 840, 330, vertical=True, tracking=0.18),
    ], stagger=0.08, fade=0.8))

    # 3-5. 食材（縦書き＋フランス語）
    for idx, (jp, fr, num) in enumerate([
        ("黄人参", "Carotte jaune", "01"),
        ("たいら貝", "Tairagai", "02"),
        ("シナモンリーフ", "Feuille de cannelier", "03"),
    ]):
        s0, s1 = B[2 + idx], B[3 + idx]
        x = 900
        top = 250
        size = 76
        caps.append(Caption(s0 + 0.25, s1 - 0.05, [
            layout(num, "italic", 40, x, top - 40, color=GOLD),
        ], fade=0.8))
        caps.append(Caption(s0 + 0.4, s1 - 0.05, [
            layout(jp, "jp", size, x, top + 30, vertical=True, tracking=0.12),
        ], stagger=0.09, fade=0.8))
        end_y = top + 30 + len(jp) * size * 1.12 + 70
        caps.append(Caption(s0 + 0.9, s1 - 0.05, [
            layout(fr, "italic", 42, x - 10 if len(fr) < 14 else 800, end_y + 10,
                   align="center", tracking=0.04, color=GOLD),
        ], stagger=0.03, fade=0.7, drift=10))

    # 6-7. コンセプト
    caps.append(Caption(B[5] + 0.3, B[6] - 0.05, [
        layout("四季を纏う、奈良の恵み。", "jp", 50, W / 2, 380, tracking=0.22),
    ], stagger=0.06, fade=0.8))
    caps.append(Caption(B[6] + 0.3, B[7] - 0.05, [
        layout("フランスの感性で、", "jp", 50, W / 2, 330, tracking=0.22),
        layout("ひと皿に。", "jp", 50, W / 2, 420, tracking=0.22),
    ], stagger=0.06, fade=0.8))

    # 8. エンドカード（縦書きタイトル）
    t8 = B[7] + 0.6
    caps.append(Caption(t8, 29.9, [
        layout("ある日のディナーより", "jp_light", 38, 740, 360, vertical=True, tracking=0.25, color=GOLD),
    ], stagger=0.05, fade=0.8, drift=10))
    caps.append(Caption(t8 + 0.7, 29.9, [
        layout("「黄人参、", "jp", 70, 620, 420, vertical=True, tracking=0.1),
        layout("たいら貝、", "jp", 70, 510, 497, vertical=True, tracking=0.1),
        layout("シナモンリーフ」", "jp", 70, 400, 497, vertical=True, tracking=0.1),
    ], stagger=0.07, fade=0.8))
    caps.append(Caption(t8 + 2.0, 29.9, [
        layout("L'Artisan KANOYA", "latin", 60, W / 2, 1560, tracking=0.08),
    ], stagger=0.035, fade=0.8))
    lines.append((t8 + 2.2, 29.9, W / 2, 1600, 180, False))
    caps.append(Caption(t8 + 2.6, 29.9, [
        layout("lartisankanoya.com", "italic", 30, W / 2, 1660, tracking=0.12, color=GOLD),
    ], stagger=0.02, fade=0.7, drift=8))
    return caps, lines


# ---------------------------------------------------------------- frame
_VIGNETTE = None
_GRAIN = None
_CAPTIONS = None


def init_worker(font_dir):
    global _VIGNETTE, _GRAIN, _CAPTIONS
    Fonts.dir = font_dir
    load_images()
    nx = (_XX - W / 2) / (W / 2)
    ny = (_YY - H / 2) / (H / 2)
    r = np.sqrt(nx ** 2 * 0.9 + ny ** 2 * 0.55)
    _VIGNETTE = (1 - 0.42 * smoothstep(0.55, 1.35, r))[..., None].astype(np.float32)
    rng = np.random.default_rng(7)
    _GRAIN = [rng.normal(0, 4.0, (H, W, 1)).astype(np.float32) for _ in range(6)]
    _CAPTIONS = build_captions()


def text_scrim(frame, t):
    """テロップの可読性のため、上部/下部をほんのり暗くする"""
    B = BOUNDS
    top = np.clip(1 - _YY / 1100, 0, 1)[..., None] ** 1.6
    bottom = np.clip((_YY - 1150) / 770, 0, 1)[..., None] ** 1.6
    k_top = 0.0
    for s0, s1 in [(B[1], B[2]), (B[2], B[3]), (B[3], B[4]), (B[4], B[5]), (B[5], B[6]), (B[6], B[7])]:
        k_top = max(k_top, ease((t - s0 + 0.2) / 0.6) * (1 - ease((t - s1 + 0.3) / 0.6)))
    k_bottom = 1 - ease((t - B[1] + 0.4) / 0.6)
    return frame * (1 - 0.5 * k_top * top) * (1 - 0.55 * k_bottom * bottom)


def render_frame(n):
    t = n / FPS
    # どのショット / トランジションか
    frame = None
    for i in range(len(SHOTS) - 1):
        b = BOUNDS[i + 1]
        if b - TR <= t < b + TR:
            frame = transition(TRANSITIONS[i], i, t, (t - (b - TR)) / (2 * TR))
            break
    if frame is None:
        i = max(k for k in range(len(SHOTS)) if BOUNDS[k] <= t)
        frame = render_shot(i, t)

    frame = text_scrim(frame, t)

    # エンドカード：画面を落としてタイトルを浮かび上がらせる
    t8 = BOUNDS[7]
    if t > t8 + 0.3:
        k = ease((t - t8 - 0.3) / 1.6)
        frame = blur(frame, 16 * k) * (1 - 0.66 * k)

    frame = frame * _VIGNETTE
    frame = frame + _GRAIN[n % len(_GRAIN)]

    caps, lines = _CAPTIONS
    draw_captions(frame, t, caps, lines)

    # 全体のフェードイン / アウト
    fade = ease(t / 1.3) * (1 - ease((t - (DURATION - 0.9)) / 0.9))
    frame = frame * fade
    return np.clip(frame, 0, 255).astype(np.uint8).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "video", "reel.mp4"))
    ap.add_argument("--stills", help="確認用: カンマ区切りの秒数で PNG を書き出す")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--music", help="BGM（先頭から30秒を使用し、終わりをフェードアウト）")
    args = ap.parse_args()

    if args.stills:
        init_worker(args.fonts)
        for s in args.stills.split(","):
            n = int(float(s) * FPS)
            img = Image.frombytes("RGB", (W, H), render_frame(n))
            path = os.path.splitext(args.out)[0] + f"_{float(s):05.1f}s.jpg"
            img.save(path, quality=88)
            print(path)
        return

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           *(["-i", args.music] if args.music else
             ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]),
           "-map", "0:v", "-map", "1:a", "-t", str(DURATION),
           "-af", f"atrim=0:{DURATION},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.3,"
                  f"afade=t=out:st={DURATION - 2.6}:d=2.6,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000",
           "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-maxrate", "10M", "-bufsize", "20M", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    with Pool(args.workers, initializer=init_worker, initargs=(args.fonts,)) as pool:
        for k, data in enumerate(pool.imap(render_frame, range(NFRAMES), chunksize=4)):
            proc.stdin.write(data)
            if k % 60 == 0:
                print(f"{k}/{NFRAMES}", file=sys.stderr, flush=True)
    proc.stdin.close()
    proc.wait()
    print(args.out)


if __name__ == "__main__":
    main()
