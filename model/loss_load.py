import torch
import torch.nn as nn


def loss_load(config, device=None):
    criteria = None
    if config["MODEL"]["CRITERIA"] == "CrossEntropy":
        criteria = torch.nn.CrossEntropyLoss()

    elif config["MODEL"]["CRITERIA"] == "FocalLoss":
        criteria = FocalLoss()

    elif config["MODEL"]["CRITERIA"] == "DiceLoss":
        criteria = DiceLoss()

    else:
        pass

    return criteria.to(device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, preds, y_true):
        assert preds.size() == y_true.size()

        # iflat = preds.contiguous().view(-1)
        # tflat = y_true.contiguous().view(-1)
        iflat = preds.view(-1)
        tflat = y_true.view(-1)

        intersection = torch.dot(iflat, tflat)

        preds_sum = torch.sum(iflat)
        true_sum = torch.sum(tflat)

        dsc = (2. * intersection + self.smooth) / (preds_sum + true_sum + self.smooth)
        dice_loss = 1 - dsc

        # weights = torch.FloatTensor([1.6]).to(self.device)
        # bce_loss = F.binary_cross_entropy(iflat, tflat, reduction='mean', weight=weights)

        return dice_loss
