import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from .config import DATA_ROOT, BATCH_SIZE, NUM_WORKERS, NUM_CLASSES, MEAN, STD, SEED

class FixedCorruption(Dataset):
    def __init__(self, base, raw_test, kind, seed=SEED): self.base,self.raw,self.kind,self.seed=base,raw_test,kind,seed
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        image,label=self.base[i]
        if self.kind=="noise":
            g=torch.Generator().manual_seed(self.seed+i); image=torch.clamp(image+torch.randn(image.shape,generator=g)*0.20,-3,3)
        elif self.kind=="rotation":
            from PIL import Image
            image=transforms.ToTensor()(transforms.functional.rotate(Image.fromarray(self.raw.data[i]), 15 if i%2==0 else -15))
            image=transforms.Normalize(MEAN,STD)(image)
        return image,label

def balanced_split(targets, fraction=0.1, seed=SEED):
    rng=np.random.default_rng(seed); tr=[]; va=[]; targets=np.asarray(targets)
    for c in range(NUM_CLASSES):
        ids=np.flatnonzero(targets==c); rng.shuffle(ids); n=int(len(ids)*fraction); va+=ids[:n].tolist(); tr+=ids[n:].tolist()
    return sorted(tr),sorted(va)

def build_data():
    basic=transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    augmented=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    raw_train=datasets.CIFAR10(str(DATA_ROOT),train=True,download=True); raw_test=datasets.CIFAR10(str(DATA_ROOT),train=False,download=True)
    tr,va=balanced_split(raw_train.targets)
    train_basic=Subset(datasets.CIFAR10(str(DATA_ROOT),train=True,transform=basic,download=True),tr)
    train_aug=Subset(datasets.CIFAR10(str(DATA_ROOT),train=True,transform=augmented,download=True),tr)
    val=Subset(datasets.CIFAR10(str(DATA_ROOT),train=True,transform=basic,download=True),va)
    test=datasets.CIFAR10(str(DATA_ROOT),train=False,transform=basic,download=True)
    loaders=lambda ds,sh: DataLoader(ds,batch_size=BATCH_SIZE,shuffle=sh,num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available())
    return {"raw_train":raw_train,"raw_test":raw_test,"train_basic":loaders(train_basic,True),"train_aug":loaders(train_aug,True),"val":loaders(val,False),"test":loaders(test,False),"noise":loaders(FixedCorruption(test,raw_test,"noise"),False),"rotation":loaders(FixedCorruption(test,raw_test,"rotation"),False)}
