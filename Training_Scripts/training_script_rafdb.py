import os, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
from sklearn.metrics import classification_report
from torch.amp import GradScaler
from torch.cuda.amp import autocast
from tqdm import tqdm
import matplotlib.pyplot as plt


def main() -> None:
    SEED = 42
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DATA_ROOT   = Path(__file__).parent / "DATASET"
    IMG_SIZE    = 224
    BATCH_SIZE  = 128
    EPOCHS      = 30
    LR          = 3e-4
    NUM_CLASSES = 7
    PATIENCE    = 5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_folder = datasets.ImageFolder(DATA_ROOT / "train", transform=train_tfms)
    test_folder  = datasets.ImageFolder(DATA_ROOT / "test",  transform=train_tfms)
    full_ds = ConcatDataset([train_folder, test_folder])

    n_total = len(full_ds)
    n_train = int(0.6 * n_total)
    n_val   = int(0.2 * n_total)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    val_ds.dataset.transform  = test_tfms
    test_ds.dataset.transform = test_tfms

    targets      = [full_ds[i][1] for i in train_ds.indices]
    class_counts = torch.bincount(torch.tensor(targets), minlength=NUM_CLASSES)
    inv_freq     = 1. / class_counts.float()
    weights      = [inv_freq[t] for t in targets]
    sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

    model     = timm.create_model('mobilenetv3_large_100', pretrained=True,
                                  num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = GradScaler(device="cuda")

    train_losses, val_losses = [], []
    train_accs, val_accs     = [], []

    best_val = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        running_loss = correct = seen = 0

        pbar = tqdm(train_ld, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                preds = model(xb)
                loss  = criterion(preds, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * xb.size(0)
            correct      += (preds.argmax(1) == yb).sum().item()
            seen         += yb.size(0)

            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             acc =f"{100*correct/seen:5.2f}%")

        train_loss = running_loss / seen
        train_acc  = 100 * correct / seen
        epoch_time = time.time() - start

        model.eval()
        val_loss = correct_val = seen_val = 0
        with torch.no_grad(), autocast():
            for xb, yb in val_ld:
                xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                preds = model(xb)
                loss  = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                correct_val += (preds.argmax(1) == yb).sum().item()
                seen_val += yb.size(0)
        val_loss /= seen_val
        val_acc  = 100 * correct_val / seen_val
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.3f}  acc={train_acc:5.2f}% | "
              f"val_loss={val_loss:.3f}  val_acc={val_acc:5.2f}% | "
              f"time={epoch_time:.1f}s")

        if val_loss < best_val:
            torch.save(model.state_dict(), "best_mobilenetv3_rafdb.pth")
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered!")
                break

    model.load_state_dict(torch.load("best_mobilenetv3_rafdb.pth"))
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad(), autocast():
        for xb, yb in tqdm(test_ld, desc="Testing", leave=False):
            xb = xb.to(DEVICE, non_blocking=True)
            preds = model(xb).argmax(1).cpu()
            y_true.extend(yb.numpy())
            y_pred.extend(preds.numpy())

    print(classification_report(
        y_true, y_pred, target_names=train_folder.classes, digits=3))

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
    torch.onnx.export(
        model, dummy, "mobilenetv3_rafdb.onnx",
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=12
    )
    print("ONNX model saved ➜ mobilenetv3_rafdb.onnx")

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses,   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs. Validation Loss')
    plt.savefig('loss_curve.png')

    plt.figure()
    plt.plot(epochs_range, train_accs, label='Training Accuracy')
    plt.plot(epochs_range, val_accs,   label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training vs. Validation Accuracy')
    plt.savefig('accuracy_curve.png')

    print("Plots saved ➜ loss_curve.png, accuracy_curve.png")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
