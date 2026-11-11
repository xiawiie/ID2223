import torch
import torch.nn as nn

def fgsm(model,image,label,device,epsilon=0.05):
    model.eval(); x=image.clone().detach().to(device).requires_grad_(True); y=label.to(device); loss=nn.CrossEntropyLoss()(model(x),y); model.zero_grad(set_to_none=True); loss.backward(); return (x+epsilon*x.grad.sign()).detach()
