import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.stats import entropy
import numpy as np

def mu_alea_epis(model, inputs):
    """predict semantic segmentation of inputs and compute aleatoric and episdemic uncertainty

    Args:
        model (Module): A pre-trained model we will use to predict
        inputs (tensor): test images, of shape (n, 3, H, W)

    Returns:
        Tuple: (mu, aleatoric, episdemic)
    """
    model.train()
    times = 50
    with torch.no_grad():
        preds = []
        for t in range(times):
            preds.append(F.softmax(model(inputs), dim=1).cpu().numpy())
        preds = np.array(preds)  # of shape (t, n, num_classes, H, W)

        mu = preds.mean(axis=0).squeeze()  # of shape (n, num_classes, H, W)
        aleatoric = entropy(mu, axis=1)  # of shape (n, H, W)
        episdemic = preds.std(axis=0).squeeze().mean(
            axis=1).squeeze()  # of shape (n, H, W)
        mu = mu.argmax(axis=1)

        return mu, aleatoric, episdemic