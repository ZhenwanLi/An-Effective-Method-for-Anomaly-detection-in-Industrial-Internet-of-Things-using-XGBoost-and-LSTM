import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, input, target):
        return F.cross_entropy(input, target, reduction=self.reduction)


class ClassBalancedLoss(nn.Module):
    def __init__(self, beta: float = 0.99, num_classes: int = 2):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, output, target):
        class_count = torch.bincount(target, minlength=self.num_classes)
        class_freq = class_count.float() / len(target)

        effective_num = 1.0 - torch.pow(self.beta, class_freq)
        class_weights = (1.0 - self.beta) / effective_num
        loss = nn.CrossEntropyLoss(weight=class_weights)(output, target)
        return loss


class WeightClassBalancedLoss(nn.Module):
    def __init__(self, beta: float = 0.99, num_classes: int = 2):
        super(WeightClassBalancedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, output, target):
        class_count = torch.bincount(target, minlength=self.num_classes)
        class_freq = class_count.float() / len(target)

        effective_num = 1.0 - torch.pow(self.beta, class_freq)
        class_weights = (1.0 - self.beta) / effective_num
        # print(f"class_weights 1: {class_weights}")

        class_weights[1:] /= 1.3
        # print(f"class_weights 2: {class_weights}")
        # class_weights /= class_weights.sum()
        # print(f"class_weights 3: {class_weights}")

        loss = nn.CrossEntropyLoss(weight=class_weights)(output, target)
        return loss
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = torch.tensor(alpha) if alpha is not None else torch.ones(self.num_classes)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha.gather(0, target.long())
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Dice Loss (Multi-class)
class MultiDiceLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-7):
        super(MultiDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, true):
        probs = F.softmax(logits, dim=1)
        loss = 0.0
        for i in range(self.num_classes):
            prob = probs[:, i]
            t = (true == i).float()
            intersection = torch.sum(prob * t)
            union = torch.sum(prob) + torch.sum(t) + self.eps
            dice = 2.0 * intersection / union
            loss += 1.0 - dice
        return loss / self.num_classes


# Tversky Loss (Multi-class)
class MultiTverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.7, beta=0.3, eps=1e-7):
        super(MultiTverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits, true):
        probs = F.softmax(logits, dim=1)
        loss = 0.0
        for i in range(self.num_classes):
            prob = probs[:, i]
            t = (true == i).float()
            tp = torch.sum(prob * t)
            fp = torch.sum(prob * (1 - t))
            fn = torch.sum((1 - prob) * t)
            tversky_index = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
            loss += 1.0 - tversky_index
        return loss / self.num_classes

