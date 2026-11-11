import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from .config import LEARNING_RATE, MOMENTUM, NUM_EPOCHS

def evaluate(model, loader, device):
    model.eval(); loss_fn = nn.CrossEntropyLoss(); loss_sum = correct = total = 0; predictions = []; labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images); loss = loss_fn(logits, labels)
            loss_sum += loss.item() * len(labels); correct += (logits.argmax(1) == labels).sum().item(); total += len(labels)
            predictions.append(logits.argmax(1).cpu()); labels_all.append(labels.cpu())
    return loss_sum / total, correct / total, torch.cat(predictions).numpy(), torch.cat(labels_all).numpy()

def fit(model, train_loader, validation_loader, device, name, weight_decay=0.0, epochs=NUM_EPOCHS):
    model.to(device)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(); best_accuracy = -1.0; best_state = None; history = []; start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train(); loss_sum = correct = total = 0; epoch_start = time.perf_counter()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images); loss = loss_fn(logits, labels)
            if amp_enabled:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(labels); correct += (logits.argmax(1) == labels).sum().item(); total += len(labels)
        validation_loss, validation_accuracy, _, _ = evaluate(model, validation_loader, device); scheduler.step()
        train_accuracy = correct / total
        history.append({"epoch": epoch, "train_loss": loss_sum / total, "train_accuracy": train_accuracy, "validation_loss": validation_loss, "validation_accuracy": validation_accuracy, "seconds": time.perf_counter() - epoch_start})
        print(f"{name} epoch {epoch}/{epochs} train={train_accuracy:.3f} val={validation_accuracy:.3f}", flush=True)
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy; best_state = deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history, {"name": name, "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "training_seconds": time.perf_counter() - start, "best_validation_accuracy": best_accuracy}

def inference_seconds(model, loader, device, max_batches=20):
    model.eval(); start = time.perf_counter()
    with torch.no_grad():
        for index, (images, _) in enumerate(loader):
            if index >= max_batches: break
            _ = model(images.to(device))
    if device.type == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - start
