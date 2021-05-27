from torch.utils.data import Dataset, Subset
import glob
import cv2 as cv
import numpy as np

WIDTH = 256
HEIGHT = 256


class DatasetTiles(Dataset):
    r"""
    Class to load the dataset of a specific defect.
    """

    def __init__(self, parent_dir, image_dir):
        r"""
        Load the dataset.
        :param parent_dir: root folder.
        :param image_dir: directory of the defect.
        """
        self.image_list = glob.glob(parent_dir + '/' + image_dir + '/Imgs/*')
        self.image_list.sort()
        print(f"{image_dir} loaded!")

    def __getitem__(self, index):
        r"""
        Get the image and its mask
        :param index: index of the specific image
        """

        X = preprocessing(self.image_list[index], False)
        y = preprocessing(self.image_list[index + 1], True)

        return X, y

    def __len__(self):
        return len(self.image_list)


def preprocessing(img, convert_to_gray):
    r"""
    Performs a preprocessing on the image.
    :param img: img to be processed
    :param convert_to_gray: true if the conversion is grayscale, false otherwise.
    """

    img = cv.imread(img)
    if convert_to_gray:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (WIDTH, HEIGHT))

    return img


def train_test_split(dataset):
    r"""
    Slip dataset in training, validation and test:
        - 70% training;
        - 20% validation:
        - 10% test.

    :param dataset: dataset to split
    :return : training, validation and test set.
    """
    length_dataset = len(dataset)

    length_train = np.int(length_dataset * 0.7)
    length_validate = np.int(length_dataset * 0.2)

    training_dataset = Subset(dataset, range(0, length_train))
    validation_dataset = Subset(dataset, range(length_train, length_train + length_validate))
    test_dataset = Subset(dataset, range(length_train + length_validate, len(dataset)))

    return training_dataset, validation_dataset, test_dataset
