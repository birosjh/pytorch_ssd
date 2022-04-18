from models.loss.confidence import confidence_loss
from models.loss.localization import localization_loss

class MultiBoxLoss():

    def __init__(alpha):

        self.alpha = alpha

    def forward(predictions, targets, location, ground_truth):

        conf_loss = confidence_loss(predictions, targets)

        loc_loss = localization_loss(predictions, location, ground_truth)

        return conf_loss + self.alpha * loc_loss

