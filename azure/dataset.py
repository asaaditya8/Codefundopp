from sklearn.utils import class_weight
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import os
import numpy as np


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
        tpath = os.path.join(self.root, 'test')
        cls = sorted(os.listdir(tpath))

        y = []
        for i, c in enumerate(cls):
            fpath = os.path.join(self.root, 'test', c)
            y.extend([i]*len(os.listdir(fpath)))

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
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(hue=0.1),
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