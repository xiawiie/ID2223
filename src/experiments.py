import json
from pathlib import Path
import pandas as pd
import torch
from .config import SEED, NUM_EPOCHS, OUTPUT_ROOT, CLASS_NAMES
from .reproducibility import set_seed
from .models import CIFAR10AlexNet, CIFAR10ResNet
from .training import fit, evaluate, inference_seconds
from .robustness import compare
from .adversarial import fgsm
from .data import build_data
from .evaluation import report

def run_all():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_data()
    results, histories, models, predictions = [], {}, {}, {}
    OUTPUT_ROOT.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.joinpath("checkpoints").mkdir(parents=True, exist_ok=True)

    def run(name, model, loader, weight_decay=0.0):
        set_seed(SEED)
        trained, history, info = fit(model, loader, data["val"], device, name, weight_decay, NUM_EPOCHS)
        test_loss, test_accuracy, pred, truth = evaluate(trained, data["test"], device)
        info.update({"test_loss": test_loss, "test_accuracy": test_accuracy,
                     "inference_seconds": inference_seconds(trained, data["test"], device),
                     "weight_decay": weight_decay})
        results.append(info); histories[name] = history; models[name] = trained
        predictions[name] = {"pred": pred.tolist(), "truth": truth.tolist()}
        torch.save(trained.state_dict(), OUTPUT_ROOT / "checkpoints" / (name.replace(" ", "_") + ".pt"))
        return trained

    alex = run("AlexNet baseline", CIFAR10AlexNet(), data["train_basic"])
    base = run("ResNet baseline", CIFAR10ResNet(), data["train_basic"])
    run("Small ResNet", CIFAR10ResNet(base_width=16), data["train_basic"])
    run("ResNet dropout 0.3", CIFAR10ResNet(dropout=0.3), data["train_basic"])
    run("ResNet dropout 0.5", CIFAR10ResNet(dropout=0.5), data["train_basic"])
    run("ResNet L2", CIFAR10ResNet(), data["train_basic"], 1e-4)
    augmented = run("ResNet augmented", CIFAR10ResNet(), data["train_aug"])

    model_df = pd.DataFrame(results)
    robust_df = pd.DataFrame(compare(base, augmented, data, device))
    report_text, confusion = report(torch.tensor(predictions["ResNet augmented"]["pred"]).numpy(), torch.tensor(predictions["ResNet augmented"]["truth"]).numpy(), CLASS_NAMES)
    # Generate one real FGSM example against a correctly classified augmented model input.
    augmented.eval(); images, labels = next(iter(data["test"]));
    with torch.no_grad():
        original_prediction = augmented(images.to(device)).argmax(1).cpu()
    correct = (original_prediction == labels).nonzero(as_tuple=False).flatten()
    index = int(correct[0]) if len(correct) else 0
    adv = fgsm(augmented, images[index:index+1], labels[index:index+1], device, epsilon=0.05)
    with torch.no_grad():
        adv_prediction = augmented(adv).argmax(1).item()
    adv_info = {"method": "FGSM", "epsilon_normalized": 0.05,
                "true_label": int(labels[index]), "original_prediction": int(original_prediction[index]),
                "adversarial_prediction": int(adv_prediction),
                "attack_success": bool(int(original_prediction[index]) != int(adv_prediction))}

    model_df.to_csv(OUTPUT_ROOT / "metrics" / "model_results.csv", index=False)
    robust_df.to_csv(OUTPUT_ROOT / "metrics" / "robustness_results.csv", index=False)
    (OUTPUT_ROOT / "metrics" / "histories.json").write_text(json.dumps(histories), encoding="utf-8")
    (OUTPUT_ROOT / "metrics" / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    (OUTPUT_ROOT / "metrics" / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (OUTPUT_ROOT / "metrics" / "confusion_matrix.csv").write_text(pd.DataFrame(confusion, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv())
    (OUTPUT_ROOT / "metrics" / "adversarial_result.json").write_text(json.dumps(adv_info, indent=2), encoding="utf-8")
    print(model_df.to_string(index=False)); print(robust_df.to_string(index=False)); print(adv_info)
    return {"device": str(device), "models": model_df, "robustness": robust_df, "adversarial": adv_info}
