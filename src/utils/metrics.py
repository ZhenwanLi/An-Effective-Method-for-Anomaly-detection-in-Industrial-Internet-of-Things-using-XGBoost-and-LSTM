from typing import Optional, Literal
import torch
from torch import Tensor
from torchmetrics.classification import MulticlassStatScores
from torchmetrics.utilities.compute import _safe_divide


class MulticlassFar(MulticlassStatScores):
    """
    Computes Far for multiclass tasks.
    
    Args:
        num_classes (int): Number of classes.
        far (Optional[Literal["micro", "macro", "weighted", "none"]]): Defines the reduction method.
        top_k (Optional[int]): Number of highest probability or logit score predictions considered.
        multidim_far (Literal["global", "samplewise"]): Defines how additional dimensions should be handled.
        ignore_index (int): Specifies a target value that is ignored.
        validate_args (bool): Indicates if input arguments and tensors should be validated.
        
    Returns:
        Tensor: A tensor with the Far score.
    """
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        print(f"tp: {tp}")
        print(f"fp: {fp}")
        print(f"tn: {tn}")
        print(f"fn: {fn}")
        return _far_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average)


def _far_reduce(
        tp: Tensor,
        fp: Tensor,
        tn: Tensor,
        fn: Tensor,
        average: Optional[Literal["binary", "micro", "macro", "weighted", "none"]],
        multidim_average: Literal["global", "samplewise"] = "global",
) -> Tensor:

    """Reduce classification statistics into Far score."""
    if average == "binary":
        return _safe_divide(fp, fp + tn)
    elif average == "micro":
        fp = fp.sum(dim=0 if multidim_average == "global" else 1)
        tn = tn.sum(dim=0 if multidim_average == "global" else 1)
        return _safe_divide(fp, fp + tn)
    else:
        score = _safe_divide(fp, fp + tn)
        if average is None or average == "none":
            return score
        if average == "weighted":
            weights = fp + tn
        else:
            weights = torch.ones_like(score)
        return _safe_divide(weights * score, weights.sum(-1, keepdim=True)).sum(-1)

