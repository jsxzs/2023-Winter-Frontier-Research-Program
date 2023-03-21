import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from .miou import *

def check_accuracy(loader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()  # set model to evaluation mode
    miou = IOUMetric(model.num_classes) # create a IOUMetric instance
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            # y = y.to(device=device)
            scores = model(x)  # of shape (bs, num_classes, H, W)
            _, preds = scores.max(1)
            # preds = preds.to(device=device)
            # the size of preds is (bs, H, W)
            # _, trues = y.max(1)
            miou.add_batch(preds.cpu().numpy(), y.cpu().numpy())
                
        return miou.evaluate()

def train(model,
          optimizer,
          loader_train,
          loader_val,
          epochs=1,
          verbose=True,
          print_every=100):
    """Train a model.

    Args:
        model (Module): A PyTorch Module giving the model to train.
        optimizer (Optimizer): An Optimizer object we will use to train the model.
        loader_train (Dataloader): A Dataloader loading the training dataset.
        loader_val (Dataloader): A Dataloader loading the validation dataset.
        epochs (int, optional): A Python integer giving the number of epochs to train for. Defaults to 1.
        verbose (bool, optional): if set to false then no output will be printed during training. Defaults to True.
        print_every (int, optional): training losses will be printed every print_every iterations. Defaults to 100.

    Returns:
        Tuple: (loss_history, train_acc_history, val_acc_history)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_history, train_acc_history, val_acc_history = [], [], []
    model = model.to(device=device)
    
    # define the loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    start_time = datetime.now()
    for e in range(epochs):
        epoch_loss = 0
        
        # Check train and val accuracy on the first iteration and at the end of each epoch.
        train_rslt = check_accuracy(loader_train, model)
        val_rslt = check_accuracy(loader_val, model)
        train_acc_history.append(train_rslt['acc'])
        val_acc_history.append(val_rslt['acc'])
        if verbose:
            print("(Epoch %d / %d) train_acc = %.4f%%, val_acc = %.4f%%, train_miou = %.4f, val_miou = %.4f" %
                  (e, epochs, 100 * train_rslt['acc'], 100 * val_rslt['acc'], train_rslt['miou'], val_rslt['miou']))

        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            # scores = torch.sigmoid(scores)
            loss = criterion(scores, y)

            # Zero out all of the gradients for the variables which the optimizer will update.
            optimizer.zero_grad()

            loss.backward()  # backwards pass
            optimizer.step()  # update the parameters of the model
            epoch_loss += loss.item()
            # loss_history.append(loss.item())

            if verbose and (t+1) % print_every == 0:
                print('(Iteration %d) loss = %.4f' % (t+1, loss.item()))
                
        loss_history.append(epoch_loss)

    # Check train and val accuracy on the last iteration.
    train_rslt = check_accuracy(loader_train, model)
    val_rslt = check_accuracy(loader_val, model)
    train_acc_history.append(train_rslt['acc'])
    val_acc_history.append(val_rslt['acc'])
    if verbose:
        print("(Epoch %d / %d) train_acc = %.4f%%, val_acc = %.4f%%, train_miou = %.4f, val_miou = %.4f" %
                (e+1, epochs, 100 * train_rslt['acc'], 100 * val_rslt['acc'], train_rslt['miou'], val_rslt['miou']))

    # print the training time
    end_time = datetime.now()
    h, remainder = divmod((end_time - start_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %d:%d:%d" % (h, m, s)
    print()
    print("training time:", time_str)
    
    return loss_history, train_acc_history, val_acc_history
