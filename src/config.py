from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("CIFAR10_DATA_ROOT", PROJECT_ROOT.parent / "data"))
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SEED = 2223
BATCH_SIZE = 512
NUM_EPOCHS = 30
LEARNING_RATE = 0.05
MOMENTUM = 0.9
NUM_WORKERS = 0
NUM_CLASSES = 10
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

SEED_STABILITY = [2223, 2224, 2225]




