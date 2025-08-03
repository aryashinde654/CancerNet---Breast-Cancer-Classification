# predict placeholder
import os
import argparse
import csv
from typing import List
import numpy as np
import tensorflow as tf

from config import Config

def collect_inputs(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for root, _, fs in os.walk(path):
            for f in fs:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    files.append(os.path.join(root, f))
        return sorted(files)
    else:
        return [path]

def load_and_preprocess(path, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_size, image_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img

def main(args):
    cfg = Config()
    model = tf.keras.models.load_model(args.weights)
    inputs = collect_inputs(args.input)

    os.makedirs("outputs", exist_ok=True)
    out_csv = os.path.join("outputs", "predictions.csv")
    print(f"[INFO] Writing predictions to {out_csv}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "pred_class", "prob_non_idc", "prob_idc"])

        for p in inputs:
            img = load_and_preprocess(p, cfg.image_size)
            img = tf.expand_dims(img, axis=0)  # [1, H, W, C]
            probs = model.predict(img, verbose=0)[0]
            pred = int(np.argmax(probs))
            writer.writerow([p, pred, float(probs[0]), float(probs[1])])
            print(f"{p} -> class={pred} (Non-IDC prob={probs[0]:.4f}, IDC prob={probs[1]:.4f})")

    print("[DONE]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to an image file or folder")
    ap.add_argument("--weights", required=True, help="Path to model .h5 file")
    args = ap.parse_args()
    main(args)
