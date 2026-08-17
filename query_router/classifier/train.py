"""Training entrypoint for the query-complexity classifier — DESIGN.md §6."""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .data import QueryComplexityDataset, load_splits
from .evaluate import adjacent_accuracy, exact_accuracy
from .losses import CostSensitiveCrossEntropy
from .model import build_model_and_tokenizer


def run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.set_grad_enabled(is_train):
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        all_preds.extend(outputs.logits.argmax(dim=-1).tolist())
        all_labels.extend(labels.tolist())
    avg_loss = total_loss / len(all_labels)
    return avg_loss, exact_accuracy(all_preds, all_labels), adjacent_accuracy(all_preds, all_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_dataset.jsonl")
    parser.add_argument("--output", default="models/classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lambda-dist", type=float, default=0.5)
    parser.add_argument("--over-penalty", type=float, default=1.5)
    parser.add_argument("--under-penalty", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_model_and_tokenizer()
    model.to(device)

    splits = load_splits(args.data)
    train_loader = DataLoader(
        QueryComplexityDataset(splits["train"], tokenizer), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(QueryComplexityDataset(splits["val"], tokenizer), batch_size=args.batch_size)

    loss_fn = CostSensitiveCrossEntropy(
        lambda_dist=args.lambda_dist, over_penalty=args.over_penalty, under_penalty=args.under_penalty
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs(args.output, exist_ok=True)
    # val_adjacent_accuracy saturates at 1.0 almost immediately with only 4 classes
    # and tolerance=1, so it can't discriminate between checkpoints. Select on
    # val_loss instead, with early stopping once it stops improving.
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_adj = run_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, val_adj = run_epoch(model, val_loader, loss_fn, None, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"train_adj={train_adj:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_adj={val_adj:.3f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_adjacent_accuracy": train_adj,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_adjacent_accuracy": val_adj,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement for {args.patience} epochs)")
                break

    with open(os.path.join(args.output, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val_loss: {best_val_loss:.4f}. Saved to {args.output}")


if __name__ == "__main__":
    main()
