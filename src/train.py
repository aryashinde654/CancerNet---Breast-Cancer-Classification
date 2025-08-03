# train placeholder
import os
import argparse
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import pickle

from config import Config
from data import list_image_paths_labels, split_paths, make_tf_dataset
from model import build_cancernet

def plot_history(history, out_path):
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main(args):
    cfg = Config()
    os.makedirs(cfg.models_dir, exist_ok=True)
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    print(f"[INFO] Listing images from: {args.data_dir}")
    paths, labels = list_image_paths_labels(args.data_dir)
    print(f"[INFO] Found {len(paths)} images.")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_paths(
        paths, labels, cfg.test_size, cfg.val_size, cfg.seed
    )
    print(f"[SPLIT] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_ds = make_tf_dataset(X_train, y_train, cfg.image_size, cfg.batch_size, cfg.buffer_size, augment=True,  shuffle=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   cfg.image_size, cfg.batch_size, cfg.buffer_size, augment=False, shuffle=False)
    test_ds  = make_tf_dataset(X_test,  y_test,  cfg.image_size, cfg.batch_size, cfg.buffer_size, augment=False, shuffle=False)

    # Class weights for imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0,1]),
        y=np.array(y_train)
    )
    class_weights = {0: float(class_weights[0]), 1: float(class_weights[1])}
    print(f"[INFO] Class Weights: {class_weights}")

    # Build & compile
    model = build_cancernet(input_shape=(cfg.image_size, cfg.image_size, cfg.channels))
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Callbacks
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        cfg.best_model_path, monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=False, verbose=1
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.patience_es, restore_best_weights=True, verbose=1
    )
    rlr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=cfg.patience_rlr, min_lr=1e-6, verbose=1
    )
    tb = tf.keras.callbacks.TensorBoard(log_dir=cfg.log_dir)

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs if args.epochs else cfg.epochs,
        class_weight=class_weights,
        callbacks=[ckpt, es, rlr, tb],
        verbose=1
    )

    # Save history plot
    plot_history(history, os.path.join(cfg.outputs_dir, "history.png"))

    # Optional quick test evaluation at the end
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save splits indices for evaluation script reproducibility
    split_info = {
        "train_count": len(X_train),
        "val_count": len(X_val),
        "test_count": len(X_test)
    }
    with open(os.path.join(cfg.outputs_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to IDC_regular dataset root")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs (default from config)")
    args = ap.parse_args()
    main(args)

with open('models/history.pkl', 'wb') as f:
    pickle.dump(H.history, f)

print("[INFO] Training history saved to models/history.pkl")
