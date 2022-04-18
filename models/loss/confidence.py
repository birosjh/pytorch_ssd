from torch import Tensor
from torch.nn import CrossEntropyLoss

cross_entropy = CrossEntropyLoss()

def confidence_loss(predictions: Tensor, targets: Tensor) -> float:

    object_indices = (targets == 0).nonzero()
    background_indices = (targets != 0).nonzero()

    object_loss = cross_entropy(
        predictions[object_indices],
        targets[object_indices]
    )

    background_loss = cross_entropy(
        predictions[background_indices],
        targets[background_indices]
    )

    return -1 * object_loss - background_loss