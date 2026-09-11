import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    n = 0
    batch_loss = 0
    for i, (data, labels) in enumerate(dataloader):
        optimizer.zero_grad()

        preds = model(data)

        loss = criterion(preds, labels)
        batch_loss += loss.item()

        loss.backward()

        optimizer.step()
        n += 1

    return (1/n) * batch_loss

    
