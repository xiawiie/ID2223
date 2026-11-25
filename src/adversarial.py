import torch
import torch.nn.functional as F
from .config import MEAN, STD


def _stats(device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return mean, std


def fgsm(model, images, labels, device, epsilon=4 / 255):
    """One-step FGSM with an L-infinity budget measured in [0, 1] pixel units.

    The inputs are normalised tensors. The step is taken in pixel space and the
    result is clipped to [0, 1] before it is normalised again, so the adversarial
    image is always a valid image.
    """
    model.eval()
    mean, std = _stats(device)
    x = images.clone().detach().to(device).requires_grad_(True)
    y = labels.to(device)
    loss = F.cross_entropy(model(x), y)
    grad, = torch.autograd.grad(loss, x)
    # std > 0, so the sign of the pixel-space gradient equals the sign of grad.
    pixels = x.detach() * std + mean
    adv_pixels = (pixels + epsilon * grad.sign()).clamp(0, 1)
    return ((adv_pixels - mean) / std).detach()


def to_pixels(images, device):
    mean, std = _stats(device)
    return (images.to(device) * std + mean).clamp(0, 1)


def attack_success_rate(model, loader, device, epsilon, max_images=2000):
    """Share of correctly classified test images whose label flips under FGSM."""
    model.eval()
    attacked = flipped = seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean = model(images).argmax(1)
        keep = clean == labels
        if keep.any():
            adv = fgsm(model, images[keep], labels[keep], device, epsilon)
            with torch.no_grad():
                flipped += (model(adv).argmax(1) != labels[keep]).sum().item()
            attacked += int(keep.sum())
        seen += len(labels)
        if seen >= max_images:
            break
    return {"epsilon": epsilon, "attacked": attacked, "flipped": flipped,
            "success_rate": flipped / max(attacked, 1)}
