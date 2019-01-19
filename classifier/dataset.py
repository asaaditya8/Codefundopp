from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from PIL import Image


class CustomDatasetFromCSV(Dataset):
    """
    Args:
        csv_path (string): path to csv file
        transform: pytorch transforms for transforms and tensor conversion
        target_transform: A function/transform that takes
            in the target and transforms it.

    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
    """
    def __init__(self, csv_path, transform=None, target_transform=None):
        # csv_path = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_outof_10.csv'
        self.data = pd.read_csv(csv_path)
        self.classes = ['Dust and Haze', 'Floods', 'Sea and Lake Ice',
                       'Severe Storms', 'Snow', 'Volcanoes',
                       'Water Color', 'Wildfires']
        self.class_to_idx = {k: i for i, k in enumerate(self.classes)}
        self.labels = np.asarray(self.data.iloc[:, 2])
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        target = self.labels[index]

        root = '/home/aaditya/PycharmProjects/Codefundopp/data'
        sample = Image.open(f'{root}/{self.data.iloc[index, 0]}.jpg')

        # Transform image to tensor
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Return image and the label
        return sample, target

    def __len__(self):
        return len(self.data.index)


if __name__ == "__main__":
    transformations = transforms.Compose([transforms.ToTensor()])
    csv_path = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_8_cleaned.csv'

    custom_mnist_from_csv = CustomDatasetFromCSV(csv_path, None)
