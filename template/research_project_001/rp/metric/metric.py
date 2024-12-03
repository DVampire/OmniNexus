import torch

from rp.registry import METRIC


@METRIC.register_module(force=True)
class Accuracy:
    def __init__(
        self,
        topk=(
            1,
            5,
        ),
    ):
        """
        Accuracy metric supporting top-k accuracy calculation.

        :param topk: Tuple of top-k values for which accuracy is calculated (e.g., (1, 5)).
        """
        self.topk = topk

    def __call__(self, output, target):
        """
        Calculate top-k accuracy.

        :param output: Model predictions (logits or probabilities), shape (batch_size, num_classes).
        :param target: Ground truth labels, shape (batch_size).
        :return: Dictionary with keys as "@k" and values as top-k accuracy percentages.
        """
        maxk = max(self.topk)
        batch_size = target.size(0)

        # Get the top-k predictions
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # Transpose for easier comparison
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Compute accuracy for each k in topk
        res = {}
        for k in self.topk:
            correct_k = (
                correct[:k].reshape(-1).float().sum(0)
            )  # Sum the correct predictions in top-k
            res[f'Acc@{k}'] = correct_k.mul_(
                100.0 / batch_size
            ).item()  # Convert to percentage
        return res


@METRIC.register_module(force=True)
class MSE:
    def __init__(self):
        """
        Mean Squared Error (MSE) metric.
        """
        pass

    def __call__(self, output, target):
        """
        Calculate MSE.

        :param output: Model predictions, shape (batch_size,).
        :param target: Ground truth labels, shape (batch_size,).
        :return: MSE as a float.
        """
        mse = ((output - target) ** 2).mean().item()
        return {'MSE': mse}


@METRIC.register_module(force=True)
class Precision:
    def __init__(self, average='binary'):
        """
        Precision metric.

        :param average: 'binary', 'micro', 'macro', or 'weighted'.
        """
        self.average = average

    def __call__(self, output, target):
        """
        Calculate Precision.

        :param output: Model predictions (logits or probabilities), shape (batch_size, num_classes).
        :param target: Ground truth labels, shape (batch_size).
        :return: Precision as a float.
        """
        pred = torch.argmax(output, dim=1)
        true_positive = (pred[target == pred]).size(0)
        total_predicted_positive = pred.size(0)

        precision = (
            true_positive / total_predicted_positive
            if total_predicted_positive > 0
            else 0.0
        )
        return {'Precision': precision}


@METRIC.register_module(force=True)
class Recall:
    def __init__(self, average='binary'):
        """
        Recall metric.

        :param average: 'binary', 'micro', 'macro', or 'weighted'.
        """
        self.average = average

    def __call__(self, output, target):
        """
        Calculate Recall.

        :param output: Model predictions (logits or probabilities), shape (batch_size, num_classes).
        :param target: Ground truth labels, shape (batch_size).
        :return: Recall as a float.
        """
        pred = torch.argmax(output, dim=1)
        true_positive = (pred[target == pred]).size(0)
        total_actual_positive = target.size(0)

        recall = (
            true_positive / total_actual_positive if total_actual_positive > 0 else 0.0
        )
        return {'Recall': recall}


@METRIC.register_module(force=True)
class F1Score:
    def __init__(self, average='binary'):
        """
        F1 Score metric.

        :param average: 'binary', 'micro', 'macro', or 'weighted'.
        """
        self.average = average

    def __call__(self, output, target):
        """
        Calculate F1 Score.

        :param output: Model predictions (logits or probabilities), shape (batch_size, num_classes).
        :param target: Ground truth labels, shape (batch_size).
        :return: F1 Score as a float.
        """
        pred = torch.argmax(output, dim=1)
        true_positive = (pred[target == pred]).size(0)
        total_predicted_positive = pred.size(0)
        total_actual_positive = target.size(0)

        precision = (
            true_positive / total_predicted_positive
            if total_predicted_positive > 0
            else 0.0
        )
        recall = (
            true_positive / total_actual_positive if total_actual_positive > 0 else 0.0
        )
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        return {'F1 Score': f1}
