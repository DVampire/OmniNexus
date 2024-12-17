"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_METRIC_DESCRIPTION = '''Generate a metric module for the project.
* The assistant MUST first create a metric.py file in src/metric directory. The metric.py file should contain the metric class.
* The assistant MUST create a __init__.py file in the src/metric directory to make it a package and import the metric class in the __init__.py file.

For example, the metric.py and __init__.py files should look like this:

** metric.py **
```python
import torch

from src.registry import METRIC

@METRIC.register_module(force=True)
class Accuracy:
    def __init__(
        self,
        topk=(
            1,
            5,
        ),
    ):
        """Accuracy metric supporting top-k accuracy calculation.

        :param topk: Tuple of top-k values for which accuracy is calculated (e.g., (1, 5)).
        """
        self.topk = topk

    def __call__(self, output, target):
        """Calculate top-k accuracy.

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
        """Mean Squared Error (MSE) metric."""
        pass

    def __call__(self, output, target):
        """Calculate MSE.

        :param output: Model predictions, shape (batch_size,).
        :param target: Ground truth labels, shape (batch_size,).
        :return: MSE as a float.
        """
        mse = ((output - target) ** 2).mean().item()
        return {'MSE': mse}


@METRIC.register_module(force=True)
class Precision:
    def __init__(self, average='binary'):
        """Precision metric.

        :param average: 'binary', 'micro', 'macro', or 'weighted'.
        """
        self.average = average

    def __call__(self, output, target):
        """Calculate Precision.

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
        """Recall metric.

        :param average: 'binary', 'micro', 'macro', or 'weighted'.
        """
        self.average = average

    def __call__(self, output, target):
        """Calculate Recall.

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
        """F1 Score metric.

        :param average: 'binary', 'micro', 'macro', or 'weighted'.
        """
        self.average = average

    def __call__(self, output, target):
        """Calculate F1 Score.

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
```

** __init__.py **
```python
from .metric import Accuracy, MSE, Precision, Recall, F1Score

__all__ = ['Accuracy', 'MSE', 'Precision', 'Recall', 'F1Score']
```

Note: Execute a bash command in the terminal to generate the metric module. The command should be run in the root directory where the metric module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
'''

ProjectMetricTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_metric',
        description=_METRIC_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the metric module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
