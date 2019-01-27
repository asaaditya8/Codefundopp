from sklearn.utils import class_weight
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import os
import numpy as np


# class CustomDatasetFromCSV(Dataset):
#     """
#     Args:
#         csv_path (string): path to csv file
#         transform: pytorch transforms for transforms and tensor conversion
#         target_transform: A function/transform that takes
#             in the target and transforms it.
#
#     Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#     """
#     def __init__(self, csv_path, transform=None, target_transform=None):
#         # csv_path = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_outof_10.csv'
#         self.data = pd.read_csv(csv_path)
#         self.classes = ['Dust and Haze', 'Floods', 'Sea and Lake Ice',
#                        'Severe Storms', 'Snow', 'Volcanoes',
#                        'Water Color', 'Wildfires']
#         self.c = len(self.classes)
#         self.class_to_idx = {k: i for i, k in enumerate(self.classes)}
#         self.labels = np.asarray(self.data.iloc[:, 2])
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         target = self.labels[index]
#
#         root = '/home/aaditya/PycharmProjects/Codefundopp/data'
#         sample = Image.open(f'{root}/{self.data.iloc[index, 0]}.jpg')
#
#         # Transform image to tensor
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         # Return image and the label
#         return sample, target
#
#     def __len__(self):
#         return len(self.data.index)


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


class WFDataset:
    def __init__(self, root):
        self.root = root
        # copied from pytorch github example
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        self.size = 299

    @property
    def classw(self):
        n_absent = 62
        n_present = 278
        y = [0]*n_absent + [1]*n_present
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
        # class_weights = np.array([3.6969697 , 0.57819905])
        # class_weights = np.array([1.0 , 1.0])
        return class_weights

    @property
    def train_ds(self):
        train_set = ImageFolder(os.path.join(self.root, 'train'), transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(hue=0.1),
            transforms.ToTensor(),
            self.normalize
        ]))
        return train_set

    @property
    def valid_ds(self):
        valid_set = ImageFolder(os.path.join(self.root, 'val'), transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            self.normalize
        ]))
        return valid_set

    @property
    def test_ds(self):
        test_set = ImageFolder(os.path.join(self.root, 'test'), transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            self.normalize
        ]))
        return test_set


if __name__ == "__main__":
    # transformations = transforms.Compose([transforms.ToTensor()])
    # csv_path = '/home/aaditya/PycharmProjects/Codefundopp/eo_nasa_file_url_cls_8_cleaned.csv'
    # custom_mnist_from_csv = CustomDatasetFromCSV(csv_path, None)
    # Formula for scaling is a' : a = 1 + 2^0.5 * (1 - cos x), where x is theta
    # factor = 1 + np.sqrt(2) * (1 - np.cos(np.pi/160 * deg))
    # rot_size = int(size * factor)
    # deg = 10

    root= 'data_wf/'
    ds = WFDataset(root)
    # Get a batch of training data
    # inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])
    # imshow(train_set[0][0])