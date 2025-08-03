# evaluate placeholder
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from config import Config
from data import list_image_paths_labels, split_paths, make_tf_dataset
from model import build_cancernet

def plot_confusion_matrix(cm, out_path, classes=["Non-IDC","IDC"]):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix")
    # Annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main(args):
    cfg = Config()

    paths, labels = list_image_paths_labels(args.data_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_paths(
        paths, labels, cfg.test_size, cfg.val_size, cfg.seed
    )

    test_ds = make_tf_dataset(X_test, y_test, cfg.image_size, cfg.batch_size, cfg.buffer_size, augment=False, shuffle=False)

    # Load model
    model = tf.keras.models.load_model(args.weights)
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.array(y_test)

    print("[REPORT]")
    print(classification_report(y_true, y_pred, target_names=["Non-IDC","IDC"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    plot_confusion_matrix(cm, os.path.join(cfg.outputs_dir, "confusion_matrix.png"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Path to IDC_regular dataset root")
    ap.add_argument("--weights", required=True, help="Path to model .h5 file (e.g., models/cancernet_best.h5)")
    args = ap.parse_args()
    main(args)
