# ID2223 Lab 1 – CNNs on CIFAR-10

AlexNet and ResNet trained on CIFAR-10 with PyTorch, plus experiments on model width, dropout, L2 regularisation, data augmentation, corrupted test images and FGSM adversarial examples.

## Layout

- `src/` – data loading, models, training loop, robustness and adversarial evaluation.
- `notebooks/Assignment1_Report.ipynb` – the main notebook. It contains the full source code (written to `src/` by `%%writefile` cells), runs all experiments and produces every table and figure used in the report.
- `outputs/metrics/` – CSV/JSON results from the last full run.
- `tests/test_smoke.py` – checks output shapes and parameter counts of the two baseline models.
- `run_seed_stability.py`, `run_seed_stability_resume.py` – optional repeat of the ResNet baseline/augmented runs with three seeds (not used in the report).

## Running

On Google Colab: open the notebook, choose a GPU runtime (Runtime → Change runtime type → T4 GPU) and run all cells. CIFAR-10 is downloaded automatically. A full run takes roughly 1.5–2 hours on a T4.

Locally:

```text
pip install -r requirements.txt
python run_all.py
```

`CIFAR10_DATA_ROOT` sets where the dataset is stored (default: a `data` folder next to the repository).

## Setup in short

- 45,000 / 5,000 class-balanced train/validation split from the official training set (seed 2223); the official 10,000-image test set is only used for final evaluation.
- 30 epochs per configuration, batch size 512, SGD (lr 0.05, momentum 0.9) with cosine annealing, mixed precision on CUDA. The checkpoint with the best validation accuracy is kept.
- The random seed is reset before each model is built, so configurations with the same architecture start from the same weights.

## Architectures

Both models follow the CIFAR-10 adaptations from Lab 2, Part A9, and are written from scratch (no torchvision model constructors or pretrained weights).

- AlexNet: five 3×3 convolutions with 64/192/384/256/256 channels, three 2×2 max-pooling layers, and a 4096 → 1024 → 512 → 10 classifier with dropout 0.5.
- ResNet: ResNet-18 layout with a 3×3, 64-channel stem, four stages of two basic blocks (64/128/256/512 channels), 1×1 projection shortcuts when the shape changes, global average pooling and a 512 → 10 linear head.
