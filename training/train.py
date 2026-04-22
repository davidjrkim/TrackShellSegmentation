import argparse
import os

import boto3
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.segmentation import deeplabv3_resnet50

from training.augmentation import get_train_transforms, get_val_transforms
from training.dataset import DanishOrthophotosDataset, build_splits
from training.loss import combined_loss

NUM_CLASSES = 6
PATIENCE = 8


def compute_class_weights(dataset: DanishOrthophotosDataset) -> torch.Tensor:
    counts = torch.zeros(NUM_CLASSES)
    for _, mask in dataset:
        for c in range(NUM_CLASSES):
            counts[c] += (mask == c).sum()
    total = counts.sum()
    weights = total / (NUM_CLASSES * counts.clamp(min=1))
    return weights / weights.sum() * NUM_CLASSES


def compute_miou(preds: torch.Tensor, targets: torch.Tensor) -> float:
    ious = []
    for c in range(NUM_CLASSES):
        pred_c = preds == c
        tgt_c = targets == c
        intersection = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union > 0:
            ious.append((intersection / union).item())
    return float(np.mean(ious)) if ious else 0.0


def upload_checkpoint(local_path: str, s3_output_dir: str, filename: str):
    if not s3_output_dir.startswith("s3://"):
        return
    bucket, prefix = s3_output_dir[5:].split("/", 1)
    s3 = boto3.client("s3",
        region_name=os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    s3.upload_file(local_path, bucket, f"{prefix.rstrip('/')}/{filename}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths, val_paths, _ = build_splits(args.data_dir)
    train_ds = DanishOrthophotosDataset(train_paths, transform=get_train_transforms())
    val_ds = DanishOrthophotosDataset(val_paths, transform=get_val_transforms())

    # Oversample chips with minority classes
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    class_weights = compute_class_weights(train_ds).to(device)

    model = deeplabv3_resnet50(num_classes=NUM_CLASSES)
    # Freeze first two backbone stages during warm-up (first 5 epochs)
    for name, param in model.backbone.named_parameters():
        if name.startswith("layer1") or name.startswith("layer2"):
            param.requires_grad = False
    model = model.to(device)

    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = AdamW([
        {"params": backbone_params, "lr": args.lr / 10},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = PolynomialLR(optimizer, total_iters=args.epochs, power=0.9)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    writer = SummaryWriter()

    best_miou = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after warm-up
        if epoch == 6:
            for param in model.backbone.parameters():
                param.requires_grad = True

        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(images)["out"]
                loss = combined_loss(out, masks, class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        val_miou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                out = model(images)["out"]
                preds = out.argmax(dim=1)
                val_miou += compute_miou(preds.cpu(), masks.cpu())
        val_miou /= len(val_loader)

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("mIoU/val", val_miou, epoch)
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_train_loss:.4f}  val_mIoU={val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            ckpt_path = "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            upload_checkpoint(ckpt_path, args.output_dir, f"epoch_{epoch:03d}_miou_{val_miou:.4f}.pth")
            upload_checkpoint(ckpt_path, args.output_dir, "best.pth")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    print(f"Training complete. Best val mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", required=True)
    train(parser.parse_args())
