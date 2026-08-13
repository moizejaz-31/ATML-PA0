import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=1, epochs=1, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = (device.type == 'cuda') if isinstance(device, torch.device) else ('cuda' in str(device))
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if inputs.shape[-2:] != (224, 224):
            inputs = nn.functional.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_cuda):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler is not None and use_cuda:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    use_cuda = (device.type == 'cuda') if isinstance(device, torch.device) else ('cuda' in str(device))

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if inputs.shape[-2:] != (224, 224):
                inputs = nn.functional.interpolate(inputs, size=(224, 224), mode='bilinear', align_corners=False)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=5, log_path="results/logs/metrics.csv", device="cpu"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logs = []

    device_obj = torch.device(device) if isinstance(device, str) else device
    model.to(device_obj)
    use_cuda = (device_obj.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device_obj, epoch=epoch, epochs=epochs, scaler=scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device_obj)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | Time: {elapsed:.2f}s")
        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    df = pd.DataFrame(logs)
    df.to_csv(log_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Feature-caching utilities for frozen-backbone training
# ---------------------------------------------------------------------------
# When the backbone is frozen, its output is identical every epoch.
# extract_features() runs the backbone ONCE and stores the results,
# then train_model_cached() trains only the fc head on those cached
# tensors — eliminating redundant forward/backward passes through 152
# layers and the per-batch 32x32→224x224 upscale.
# ---------------------------------------------------------------------------

def extract_features(model, loader, device):
    """
    Run the frozen backbone once over *loader* and return cached
    (features, labels) tensors stored on *device*.

    Works by temporarily replacing model.fc with an identity so we
    capture the avgpool output (the backbone feature vector).
    """
    model.eval()
    device_obj = torch.device(device) if isinstance(device, str) else device
    model.to(device_obj)
    use_cuda = (device_obj.type == 'cuda')

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    original_fc = model.fc
    model.fc = nn.Identity()          # bypass the classification head

    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting features", leave=False):
            inputs = inputs.to(device_obj, non_blocking=True)
            if inputs.shape[-2:] != (224, 224):
                inputs = nn.functional.interpolate(
                    inputs, size=(224, 224),
                    mode='bilinear', align_corners=False
                )
            with torch.cuda.amp.autocast(enabled=use_cuda):
                feats = model(inputs)             # (B, 2048)
            feats = torch.nan_to_num(feats.float(), nan=0.0, posinf=1.0, neginf=-1.0)
            all_features.append(feats)
            all_labels.append(targets.to(device_obj, non_blocking=True))


    model.fc = original_fc                    # restore the real head
    return torch.cat(all_features), torch.cat(all_labels)



def train_model_cached(model, train_loader, val_loader, criterion, optimizer,
                       epochs=5, log_path="results/logs/metrics.csv",
                       device="cpu", batch_size=256):
    """
    Drop-in replacement for train_model() when the backbone is frozen.

    1. Extract features once for train and val sets.
    2. Train only model.fc on the cached feature tensors.

    Returns a DataFrame identical to train_model().
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    model.to(device)

    # --- one-time feature extraction (the expensive part) ----------------
    print("Caching train features …")
    train_feats, train_labels = extract_features(model, train_loader, device)
    print("Caching val features …")
    val_feats, val_labels = extract_features(model, val_loader, device)

    # Build lightweight TensorDataset loaders
    from torch.utils.data import TensorDataset, DataLoader
    cached_train = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=batch_size, shuffle=True
    )
    cached_val = DataLoader(
        TensorDataset(val_feats, val_labels),
        batch_size=batch_size, shuffle=False
    )

    fc = model.fc                     # train only the head
    logs = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- train -------------------------------------------------------
        fc.train()
        running_loss = correct = total = 0
        pbar = tqdm(cached_train, desc=f"Epoch {epoch}/{epochs} [Train]",
                    leave=False)
        for feats, targets in pbar:
            optimizer.zero_grad()
            outputs = fc(feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * feats.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- validate ----------------------------------------------------
        fc.eval()
        running_loss = correct = total = 0
        with torch.no_grad():
            for feats, targets in cached_val:
                outputs = fc(feats)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * feats.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.2f}s")
        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    df = pd.DataFrame(logs)
    df.to_csv(log_path, index=False)
    return df
