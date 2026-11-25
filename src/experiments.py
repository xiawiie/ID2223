import json
import platform
import numpy as np
import pandas as pd
import torch
from .config import SEED, NUM_EPOCHS, OUTPUT_ROOT, CLASS_NAMES, BATCH_SIZE, LEARNING_RATE, MOMENTUM
from .reproducibility import set_seed
from .models import CIFAR10AlexNet, CIFAR10ResNet
from .training import fit, evaluate, benchmark_inference
from .robustness import compare
from .adversarial import fgsm, to_pixels, attack_success_rate
from .data import build_data
from .evaluation import report

EPSILONS = [1 / 255, 2 / 255, 4 / 255, 8 / 255]


def run_info(device):
    info = {"device": str(device), "torch": torch.__version__, "python": platform.python_version(),
            "seed": SEED, "epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM, "optimizer": "SGD + cosine annealing", "mixed_precision": device.type == "cuda"}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def adversarial_example(model, data, device):
    """Attack the first correctly classified test image with the smallest budget that flips it."""
    images, labels = next(iter(data["test"]))
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        clean_pred = model(images).argmax(1)
    index = int((clean_pred == labels).nonzero()[0])
    x, y = images[index:index + 1], labels[index:index + 1]
    for epsilon in EPSILONS:
        adv = fgsm(model, x, y, device, epsilon)
        with torch.no_grad():
            clean_prob = torch.softmax(model(x), 1)[0]
            adv_prob = torch.softmax(model(adv), 1)[0]
        if adv_prob.argmax().item() != y.item():
            break
    np.savez_compressed(OUTPUT_ROOT / "metrics" / "adversarial_example.npz",
                        clean=to_pixels(x, device)[0].cpu().numpy(), adversarial=to_pixels(adv, device)[0].cpu().numpy())
    return {"method": "FGSM (single step, L-infinity, pixel space)", "epsilon": epsilon,
            "epsilon_in_255": round(epsilon * 255), "test_index": index,
            "true_label": int(y), "original_prediction": int(clean_prob.argmax()),
            "original_confidence": float(clean_prob.max()),
            "adversarial_prediction": int(adv_prob.argmax()), "adversarial_confidence": float(adv_prob.max()),
            "attack_success": bool(adv_prob.argmax().item() != y.item())}


def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = OUTPUT_ROOT / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)
    (metrics / "run_info.json").write_text(json.dumps(run_info(device), indent=2), encoding="utf-8")
    set_seed(SEED)
    data = build_data()
    results, histories, models, predictions = [], {}, {}, {}

    def run(name, build_model, loader, weight_decay=0.0):
        # Seed before the model is built so that every configuration starts from
        # the same random state (same initial weights for identical architectures).
        set_seed(SEED)
        model = build_model()
        trained, history, info = fit(model, loader, data["val"], device, name, weight_decay, NUM_EPOCHS)
        test_loss, test_accuracy, pred, truth = evaluate(trained, data["test"], device)
        info.update({"test_loss": test_loss, "test_accuracy": test_accuracy, "weight_decay": weight_decay,
                     "seconds_per_epoch": info["training_seconds"] / NUM_EPOCHS})
        info.update(benchmark_inference(trained, data["test"], device))
        results.append(info); histories[name] = history; models[name] = trained
        predictions[name] = {"pred": pred.tolist(), "truth": truth.tolist()}
        torch.save(trained.state_dict(), OUTPUT_ROOT / "checkpoints" / (name.replace(" ", "_") + ".pt"))
        pd.DataFrame(results).to_csv(metrics / "model_results.csv", index=False)
        (metrics / "histories.json").write_text(json.dumps(histories), encoding="utf-8")
        return trained

    alex = run("AlexNet baseline", CIFAR10AlexNet, data["train_basic"])
    base = run("ResNet baseline", CIFAR10ResNet, data["train_basic"])
    run("ResNet width 16", lambda: CIFAR10ResNet(base_width=16), data["train_basic"])
    run("ResNet width 32", lambda: CIFAR10ResNet(base_width=32), data["train_basic"])
    run("ResNet dropout 0.3", lambda: CIFAR10ResNet(dropout=0.3), data["train_basic"])
    run("ResNet dropout 0.5", lambda: CIFAR10ResNet(dropout=0.5), data["train_basic"])
    run("ResNet L2", CIFAR10ResNet, data["train_basic"], 1e-4)
    augmented = run("ResNet augmented", CIFAR10ResNet, data["train_aug"])

    robust_rows, robust_class_rows, robust_predictions = compare(base, augmented, data, device)
    report_text, confusion = report(np.array(predictions["ResNet augmented"]["pred"]),
                                    np.array(predictions["ResNet augmented"]["truth"]), CLASS_NAMES)

    rates = []
    for name, model in [("AlexNet baseline", alex), ("ResNet baseline", base), ("ResNet augmented", augmented)]:
        for epsilon in EPSILONS:
            row = attack_success_rate(model, data["test"], device, epsilon)
            row.update({"name": name, "epsilon_in_255": round(epsilon * 255)})
            rates.append(row)
    adv_info = adversarial_example(augmented, data, device)

    model_df = pd.DataFrame(results)
    robust_df = pd.DataFrame(robust_rows)
    model_df.to_csv(metrics / "model_results.csv", index=False)
    robust_df.to_csv(metrics / "robustness_results.csv", index=False)
    pd.DataFrame(robust_class_rows).to_csv(metrics / "robustness_per_class.csv", index=False)
    pd.DataFrame(rates).to_csv(metrics / "adversarial_rates.csv", index=False)
    (metrics / "robustness_predictions.json").write_text(json.dumps(robust_predictions), encoding="utf-8")
    (metrics / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    (metrics / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (metrics / "confusion_matrix.csv").write_text(pd.DataFrame(confusion, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv())
    (metrics / "adversarial_result.json").write_text(json.dumps(adv_info, indent=2), encoding="utf-8")
    print(model_df.to_string(index=False)); print(robust_df.to_string(index=False))
    print(pd.DataFrame(rates).to_string(index=False)); print(adv_info)
    return {"device": str(device), "models": model_df, "robustness": robust_df, "adversarial": adv_info}
