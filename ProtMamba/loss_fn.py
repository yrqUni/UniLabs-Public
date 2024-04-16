import torch
import torch.nn as nn

def masked_cross_entropy_loss(predictions, targets, mask_pos):
    targets = targets.squeeze(-1)
    mask_pos = mask_pos.squeeze(-1)
    criterion = nn.CrossEntropyLoss(reduction='none')
    loss = criterion(predictions.view(-1, predictions.size(-1)), targets.view(-1))
    mask_pos = mask_pos.view(-1).float()
    masked_loss = loss * mask_pos
    final_loss = masked_loss.sum() / mask_pos.sum()
    return final_loss