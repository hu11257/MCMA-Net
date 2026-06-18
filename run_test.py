import argparse
import os

import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from script.dataloader import (
    MyLazyDataset,
    load_conservation,
    load_multiple_shapes,
    load_sequences_and_labels,
    my_custom_collate,
)
from script.model import MCMA_Net


def main():
    parser = argparse.ArgumentParser(description="Run MCMA-Net on a test dataset.")
    parser.add_argument("--data_dir", required=True, help="Path to one TF dataset folder, e.g. Dataset/example_TF")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint, e.g. checkpoints/example_TF/model_final.pth")
    parser.add_argument("--out_dir", default="result/test_run", help="Directory for prediction and metric outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_folder = os.path.join(args.data_dir, "Sequence")
    shape_folder = os.path.join(args.data_dir, "Shape")
    cons_folder = os.path.join(args.data_dir, "conservation")

    test_sequences, test_labels = load_sequences_and_labels(os.path.join(seq_folder, "Test_seq.csv"))
    test_shapes = load_multiple_shapes(shape_folder, prefix="Test")
    test_cons = load_conservation(cons_folder, prefix="Test")

    test_dataset = MyLazyDataset(test_sequences, test_labels, test_shapes, test_cons)
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=my_custom_collate,
        drop_last=False,
    )

    model = MCMA_Net().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for seq_onehot, shapes, cons, labels in test_loader:
            seq_onehot = seq_onehot.to(device)
            shapes = shapes.to(device)
            cons = cons.to(device)

            logits = model(seq_onehot, shapes, cons)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

            y_prob.extend(probs.tolist())
            y_true.extend(labels.numpy().tolist())

    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    pred_df = pd.DataFrame(
        {
            "sample_index": list(range(len(test_sequences))),
            "sequence": test_sequences,
            "true_label": y_true,
            "probability": y_prob,
            "prediction": y_pred,
        }
    )
    pred_df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

    print("Prediction file:", os.path.join(args.out_dir, "predictions.csv"))
    print("Metric file:", os.path.join(args.out_dir, "metrics.csv"))
    print(metrics)


if __name__ == "__main__":
    main()