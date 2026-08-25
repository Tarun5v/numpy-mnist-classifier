"""
Generate a demo GIF for the README.

Runs headlessly — no tkinter display needed. Uses PIL to render the canvas,
draws a clean digit via font rendering, feeds it through the trained model,
and assembles the result into an animated GIF.

Usage:
    python3 record_demo.py          # produces demo.gif in repo root
    python3 record_demo.py out.gif  # custom output path
"""

import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.insert(0, os.path.dirname(__file__))
from neural_network import NeuralNetwork

# ── layout constants ────────────────────────────────────────────────────────
CANVAS  = 280
PAD     = 16
SB_W    = 200
IMG_W   = CANVAS + PAD * 3 + SB_W
IMG_H   = CANVAS + PAD * 2 + 40
BG      = "#f0f0f0"


def _get_font(size, bold=False):
    names = (
        ["Helvetica Bold", "Helvetica"] if bold else
        ["Helvetica", "Arial", "DejaVu Sans"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ── digit drawing via font ──────────────────────────────────────────────────

def render_digit(digit, progress=1.0):
    """
    Render a single digit onto a 280x280 grayscale canvas using a large font.
    progress: 0-1, for drawing animation (reveal left-to-right via masking).
    """
    img = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(img)

    # draw the digit large and centered
    font = _get_font(220, bold=True)
    bbox = draw.textbbox((0, 0), str(digit), font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (CANVAS - tw) // 2 - bbox[0]
    y = (CANVAS - th) // 2 - bbox[1]
    draw.text((x, y), str(digit), fill=255, font=font)

    # light blur to soften edges (looks more handwritten)
    img = img.filter(ImageFilter.GaussianBlur(2.0))

    # threshold back to sharp
    img = img.point(lambda p: 255 if p > 30 else 0)

    # animation: reveal with a vertical wipe from left to right
    if progress < 1.0:
        mask = Image.new("L", (CANVAS, CANVAS), 0)
        mask_draw = ImageDraw.Draw(mask)
        cutoff = int(CANVAS * progress)
        mask_draw.rectangle([0, 0, cutoff, CANVAS], fill=255)
        img = Image.composite(img, Image.new("L", (CANVAS, CANVAS), 0), mask)

    return img


# ── sidebar rendering ───────────────────────────────────────────────────────

def draw_sidebar(draw, x0, prediction=None, probs=None):
    Y = PAD

    draw.text((x0, Y), "Prediction:", fill="#333333",
              font=_get_font(16, bold=True))
    Y += 24

    if prediction is not None:
        draw.text((x0 + 30, Y), str(prediction), fill="#0055ff",
                  font=_get_font(64, bold=True))
    Y += 80

    if probs is not None:
        conf = probs[prediction]
        draw.text((x0, Y), "Confidence:", fill="#555555", font=_get_font(14))
        Y += 20
        draw.text((x0, Y), f"{conf:.1%}", fill="#222222",
                  font=_get_font(20, bold=True))
        Y += 30
    else:
        draw.text((x0, Y), "Draw something!", fill="#555555",
                  font=_get_font(13))
        Y += 24

    for label in ["Predict", "Clear"]:
        draw.rounded_rectangle([x0, Y, x0 + SB_W - 10, Y + 30], radius=6,
                               fill="#e0e0e0", outline="#999999")
        draw.text((x0 + SB_W // 2 - 30, Y + 5), label, fill="#333333",
                  font=_get_font(14))
        Y += 40

    Y += 10
    draw.text((x0, Y), "Probabilities:", fill="#333333",
              font=_get_font(14, bold=True))
    Y += 24

    bar_max = SB_W - 50
    for d in range(10):
        draw.text((x0, Y + 2), f"{d}:", fill="#555555", font=_get_font(13))
        bx = x0 + 24
        draw.rectangle([bx, Y + 2, bx + bar_max, Y + 17],
                       fill="#e0e0e0", outline="#cccccc")
        if probs is not None:
            w = max(1, int(probs[d] * bar_max))
            color = "#0055ff" if d == prediction else "#66aaff"
            draw.rectangle([bx, Y + 2, bx + w, Y + 17], fill=color)
            draw.text((bx + bar_max + 5, Y + 2), f"{probs[d]:.0%}",
                      fill="#555555", font=_get_font(12))
        else:
            draw.text((bx + bar_max + 5, Y + 2), "0%",
                      fill="#555555", font=_get_font(12))
        Y += 20


def build_frame(canvas_img, prediction=None, probs=None):
    frame = Image.new("RGB", (IMG_W, IMG_H), BG)
    draw  = ImageDraw.Draw(frame)

    draw.text((PAD, 8), "MNIST Digit Recognizer", fill="#333333",
              font=_get_font(18, bold=True))

    draw.text((PAD, IMG_H - 24),
              "Draw with mouse  \u2022  Click Predict to classify  \u2022  Click Clear to reset",
              fill="#999999", font=_get_font(10))

    canvas_rgb = Image.merge("RGB", [canvas_img, canvas_img, canvas_img])
    cy = 38
    draw.rectangle([PAD - 2, cy - 2, PAD + CANVAS + 2, cy + CANVAS + 2],
                   outline="#888888", width=2)
    frame.paste(canvas_rgb, (PAD, cy))

    draw_sidebar(draw, PAD * 2 + CANVAS, prediction, probs)
    return frame


# ── main ────────────────────────────────────────────────────────────────────

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "demo.gif")

    print("Loading trained model...")
    model = NeuralNetwork.load_weights("models/mnist_model.npy")

    # pick a digit the model classifies correctly and confidently
    best_digit, best_conf = 7, 0
    for d in range(10):
        c = render_digit(d)
        a = np.array(c.resize((28, 28), Image.LANCZOS)).astype(np.float64) / 255.0
        p = model.predict_proba(a.reshape(1, -1))[0]
        conf = p[d]
        if conf > best_conf:
            best_digit, best_conf = d, conf
    print(f"Using digit {best_digit} (model confidence: {best_conf:.1%})")

    frames  = []
    durations = []

    def add(frame, ms=60):
        frames.append(frame.convert("RGB"))
        durations.append(ms)

    # 1 — blank canvas
    blank = Image.new("L", (CANVAS, CANVAS), 0)
    add(build_frame(blank), 1200)

    # 2 — animated wipe reveal
    print("Animating digit drawing...")
    steps = 30
    for i in range(1, steps + 1):
        canvas = render_digit(best_digit, progress=i / steps)
        add(build_frame(canvas), 40 if i < steps else 400)

    # 3 — hold completed digit
    canvas = render_digit(best_digit)
    add(build_frame(canvas), 600)

    # 4 — prediction
    print("Running prediction...")
    img_28 = canvas.resize((28, 28), Image.LANCZOS)
    arr = np.array(img_28).astype(np.float64) / 255.0
    probs = model.predict_proba(arr.reshape(1, -1))[0]
    pred = int(np.argmax(probs))
    print(f"Predicted: {pred} ({probs[pred]:.1%})")

    result_frame = build_frame(canvas, pred, probs)
    add(result_frame, 2500)

    # 5 — pulse the winning bar
    for _ in range(3):
        bright = result_frame.copy()
        d = ImageDraw.Draw(bright)
        bar_x0 = PAD * 2 + CANVAS + 24
        bar_y0 = 38 + 24 + 80 + 24 + 30 + 40 + 40 + 10 + 24 + pred * 20 + 2
        d.rectangle([bar_x0, bar_y0, bar_x0 + 138, bar_y0 + 15], fill="#0033aa")
        add(bright, 120)
        add(result_frame, 120)

    # 6 — final hold
    add(result_frame, 1500)

    # save
    print(f"Saving {len(frames)}-frame GIF to {out_path}...")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Done! {out_path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
