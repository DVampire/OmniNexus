"""This file contains the function calling implementation for different actions.

This is similar to the functionality of `CodeActResponseParser`.
"""

from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)

_DATASET_DESCRIPTION = '''Generate a dataset module for the project.
* The assistant MUST first create a dataset.py file in src/dataset directory. The dataset.py file should contain the dataset class.
* The assistant MUST create a __init__.py file in the src/dataset directory to make it a package and import the dataset class in the __init__.py file.

Take image classification as an example, the dataset.py and __init__.py files should look like this:

** dataset.py **
```python
import os

from PIL import Image
from torch.utils.data import Dataset

from src.logger import logger
from src.registry import DATASET
from src.utils import assemble_project_path

@DATASET.register_module(force=True)
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """A custom image dataset class for loading images and their corresponding labels.

        :param image_dir: Root directory of the dataset, with the structure:
            root/
            ├── class1/
            │   ├── img1.jpg
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
                ├── img1.jpg
                ├── img2.jpg
                └── ...
        :param transform: Transformations to apply to the images, such as Resize, ToTensor, etc.
        """
        self.image_dir = assemble_project_path(image_dir)
        self.transform = transform

        # Collect all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(os.listdir(self.image_dir))
        }

        logger.info(f'Found {len(self.class_to_idx)} classes in the dataset.')
        for cls_name, idx in self.class_to_idx.items():
            cls_dir = os.path.join(self.image_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(idx)

    def __len__(self):
        """Returns the total number of images in the dataset.

        :return: Integer representing the total number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Fetches the image and label at the specified index.

        :param idx: Index of the data item.
        :return: A tuple (image, label), where image is the transformed image tensor and label is an integer class index.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert('RGB')  # Ensure 3-channel RGB

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

```

** __init__.py **
```python
from .dataset import ImageDataset

__all__ = ['ImageDataset']
```

Note: Execute a bash command in the terminal to generate the dataset module. The command should be run in the root directory where the dataset module should be generated.
* Long running commands: For commands that may run indefinitely, it should be run in the background and the output should be redirected to a file, e.g. command = `python3 app.py > server.log 2>&1 &`.
* Interactive: If a bash command returns exit code `-1`, this means the process is not yet finished. The assistant must then send a second call to terminal with an empty `command` (which will retrieve any additional logs), or it can send additional text (set `command` to the text) to STDIN of the running process, or it can send command=`ctrl+c` to interrupt the process.
* Timeout: If a command execution result says "Command timed out. Sending SIGINT to the process", the assistant should retry running the command in the background.
'''

ProjectDatasetTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='project_dataset',
        description=_DATASET_DESCRIPTION,
        parameters={
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The bash command to generate the dataset module in the terminal. Can be empty to view additional logs when previous exit code is `-1`. Can be `ctrl+c` to interrupt the currently running process.',
                },
            },
            'required': ['command'],
        },
    ),
)
