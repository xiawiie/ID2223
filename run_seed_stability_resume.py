import json
from pathlib import Path
import pandas as pd
import torch
from src.config import SEED_STABILITY, NUM_EPOCHS, OUTPUT_ROOT
from src.reproducibility import set_seed
from src.data import build_data
from src.models import CIFAR10ResNet
from src.training import fit, evaluate

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path=OUTPUT_ROOT/"metrics"/"seed_stability_partial.csv"
    existing=pd.read_csv(path) if path.exists() else pd.DataFrame()
    done={(int(r.seed),r.condition) for r in existing.itertuples()} if not existing.empty else set()
    rows=existing.to_dict("records") if not existing.empty else []
    for seed in SEED_STABILITY:
        data=build_data()
        for name, loader in [("ResNet baseline",data["train_basic"]),("ResNet augmented",data["train_aug"])]:
            if (seed,name) in done: continue
            set_seed(seed); model=CIFAR10ResNet(); model,history,info=fit(model,loader,data["val"],device,f"{name} seed {seed}",0.0,NUM_EPOCHS)
            loss,accuracy,_,_=evaluate(model,data["test"],device)
            rows.append({"seed":seed,"condition":name,"best_validation_accuracy":info["best_validation_accuracy"],"test_accuracy":accuracy,"test_loss":loss,"training_seconds":info["training_seconds"]})
            pd.DataFrame(rows).to_csv(path,index=False)
    result=pd.DataFrame(rows).sort_values(["condition","seed"]); result.to_csv(OUTPUT_ROOT/"metrics"/"seed_stability.csv",index=False)
    summary=result.groupby("condition").agg(test_accuracy_mean=("test_accuracy","mean"),test_accuracy_std=("test_accuracy","std"),validation_mean=("best_validation_accuracy","mean"),validation_std=("best_validation_accuracy","std")).reset_index(); summary.to_csv(OUTPUT_ROOT/"metrics"/"seed_stability_summary.csv",index=False)
    print(result.to_string(index=False)); print(summary.to_string(index=False))
if __name__ == "__main__": main()
