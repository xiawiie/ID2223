# CIFAR-10 CNN Experiment

This project implements all mandatory Tasks 1-6 from the lab assignment using PyTorch. Source code, comments, and the final report notebook are written in English.

## Project layout

- `src/`: reusable data, model, training, evaluation, robustness, and adversarial code.
- `tests/test_smoke.py`: minimal model smoke test.
- `outputs/`: measured CSV metrics, histories, confusion matrix, predictions, adversarial result, seed-stability summary, and checkpoints.
- `notebooks/Assignment1_Report.ipynb`: executed English report and visual presentation.

## Run

From this directory:

```text
python run_all.py
```

The main experiment uses a fixed 30-epoch budget per configuration, batch size 512, CUDA mixed precision when available, and a fixed random seed. It runs AlexNet, ResNet, capacity, Dropout, L2, augmentation, corruption, prediction, and FGSM evaluations. The final report focuses on the mandatory Tasks 1-6; optional multi-seed stability analysis is not included in the final submission.

Then open `notebooks/Assignment1_Report.ipynb` and run all cells to reproduce the report from the generated outputs.

The official CIFAR-10 test set is kept separate from training and validation. Checkpoints are selected by validation accuracy before final test evaluation. The dataset is intentionally kept outside this repository. Set `CIFAR10_DATA_ROOT` to the local CIFAR-10 directory, or place the dataset in the sibling `data` directory expected by the default configuration.

## Architecture assumption

The implementation follows the official Part A9 CIFAR-10 adaptations: AlexNet uses 64/192/384/256/256 convolutional channels and a 4096 -> 1024 -> 512 -> 10 classifier; ResNet uses a 64-channel stem, four two-block stages with 64/128/256/512 channels, projection shortcuts, global average pooling, and a 512 -> 10 head. No pretrained or torchvision model constructors are used.
