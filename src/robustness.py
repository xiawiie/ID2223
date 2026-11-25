from .training import evaluate
from .config import NUM_CLASSES


def per_class(pred, truth):
    return [float((pred[truth == c] == c).mean()) for c in range(NUM_CLASSES)]


def compare(model_a, model_b, loaders, device):
    """Evaluate two models on the same clean, noisy and rotated test images."""
    rows, class_rows, predictions = [], [], {}
    for name, model in [("ResNet basic", model_a), ("ResNet augmented", model_b)]:
        row = {"name": name}
        predictions[name] = {}
        for split in ("test", "noise", "rotation"):
            _, accuracy, pred, truth = evaluate(model, loaders[split], device)
            key = "clean" if split == "test" else split
            row[f"{key}_accuracy"] = accuracy
            predictions[name][key] = pred.tolist()
            for c, value in enumerate(per_class(pred, truth)):
                class_rows.append({"name": name, "condition": key, "class": c, "accuracy": value})
        row["noise_drop"] = row["clean_accuracy"] - row["noise_accuracy"]
        row["rotation_drop"] = row["clean_accuracy"] - row["rotation_accuracy"]
        rows.append(row)
    predictions["truth"] = truth.tolist()
    return rows, class_rows, predictions
