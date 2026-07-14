"""
Train Emergency Vehicle Classifier
====================================
Run:
    python detection/train.py --data_root ./dataset/vehicles --epochs 50
"""

import argparse, os, time
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from model   import EmergencyVehicleClassifier
from dataset import get_loaders

# Fix Python 3.8 walrus operator issue
CLASSES = EmergencyVehicleClassifier.CLASSES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  type=str,   default='./dataset/vehicles')
    p.add_argument('--save_dir',   type=str,   default='./results/detection')
    p.add_argument('--epochs',     type=int,   default=50)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--image_size', type=int,   default=224)
    p.add_argument('--patience',   type=int,   default=10,
                   help='Early stopping patience')
    return p.parse_args()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    class_correct = [0] * len(CLASSES)
    class_total   = [0] * len(CLASSES)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
        for c in range(len(CLASSES)):
            mask = labels == c
            class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()

    per_class = {CLASSES[i]: (class_correct[i] / max(class_total[i], 1)) for i in range(len(CLASSES))}
    return total_loss / total, correct / total, per_class


def train():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(save_dir / 'train.log', 'w')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    train_loader, val_loader = get_loaders(
        args.data_root, args.image_size, args.batch_size)

    model     = EmergencyVehicleClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, per_cls = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        per_cls_str = '  '.join(f"{k}={v:.0%}" for k, v in per_cls.items())
        log(f"Epoch [{epoch:3d}/{args.epochs}] "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.2%} | "
            f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.2%} | "
            f"{per_cls_str} | {elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'val_acc': vl_acc},
                       save_dir / 'classifier_best.pth')
            log(f"  ✓ Best model saved (val_acc={vl_acc:.2%})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log(f"Early stopping at epoch {epoch}")
                break

    log(f"\nTraining complete. Best val accuracy: {best_val_acc:.2%}")
    log_file.close()


if __name__ == '__main__':
    # Fix import for running from project root
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    train()
