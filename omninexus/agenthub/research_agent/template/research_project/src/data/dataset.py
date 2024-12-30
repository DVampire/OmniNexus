import os

from PIL import Image
from rp.logger import research_logger as logger
from rp.registry import DATASET
from rp.utils import assemble_project_path
from torch.utils.data import Dataset


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
