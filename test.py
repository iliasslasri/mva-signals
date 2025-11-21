import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import torch.nn as nn

from dataset import SignalsDataset
from models import CNN_LSTM_SNR_Model   # adjust if you test other models


def evaluate(model, dataloader, device, loss_fn):
    model.eval()

    total_loss = 0
    correct = 0
    n = 0

    per_class_correct = {}
    per_class_total = {}

    snr_correct = {}
    snr_total = {}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels, snr in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)
            snr = snr.to(device).unsqueeze(1)

            outputs = model(signals, snr)
            loss = loss_fn(outputs, labels.long())
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            n += labels.size(0)

            # ---- Per-class accuracy ----
            for p, l in zip(preds, labels):
                l = int(l.item())
                if l not in per_class_total:
                    per_class_total[l] = 0
                    per_class_correct[l] = 0
                per_class_total[l] += 1
                per_class_correct[l] += (p == l).item()

            # ---- Per-SNR accuracy ----
            for s, p, l in zip(snr, preds, labels):
                s = int(s.item())
                if s not in snr_total:
                    snr_total[s] = 0
                    snr_correct[s] = 0
                snr_total[s] += 1
                snr_correct[s] += (p == l).item()

    avg_loss = total_loss / n
    overall_acc = correct / n

    per_class_acc = {
        c: per_class_correct[c] / per_class_total[c]
        for c in per_class_total
    }

    per_snr_acc = {
        s: snr_correct[s] / snr_total[s]
        for s in snr_total
    }

    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, overall_acc, per_class_acc, per_snr_acc, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--test_path", type=str, default="test.hdf5", help="Path to test dataset")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--transform", type=str, default=None)
    parser.add_argument("--exclude_zero_snr", action="store_true")
    parser.add_argument("--only_one_snr", type=int, default=-1)
    parser.add_argument("--include_snr", action="store_true")
    args = parser.parse_args()

    print("Loading dataset...")
    test_dataset = SignalsDataset(
        args.test_path,
        transform=args.transform,
        magnitude_only=True,
        window_size=args.window_size,
        exclude_zero_snr=args.exclude_zero_snr,
        only_one_snr=args.only_one_snr,
        augment=False,
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = CNN_LSTM_SNR_Model(
        n_classes=6,
        n_channels=2,
        hidden_size=64,
        include_snr=args.include_snr
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")

    loss_fn = nn.CrossEntropyLoss()

    print("Running evaluation...")
    avg_loss, overall_acc, per_class_acc, per_snr_acc, cm = evaluate(
        model, test_loader, device, loss_fn
    )

    print("\n==================== RESULTS ====================\n")

    print(f"Test Loss:      {avg_loss:.4f}")
    print(f"Test Accuracy:  {overall_acc*100:.2f}%\n")

    print("---- Accuracy per Class ----")
    for c in sorted(per_class_acc):
        print(f"Class {c}: {per_class_acc[c]*100:.2f}%")

    print("\n---- Accuracy per SNR ----")
    for s in sorted(per_snr_acc):
        print(f"SNR {s} dB: {per_snr_acc[s]*100:.2f}%")

    print("\n---- Confusion Matrix ----")
    print(cm)

    print("\n=================================================\n")


if __name__ == "__main__":
    main()
