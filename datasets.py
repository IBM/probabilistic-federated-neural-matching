import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, CIFAR10

class MNIST_truncated(data.Dataset):

	def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

		self.root = root
		self.dataidxs = dataidxs
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		self.download = download

		self.data, self.target = self.__build_truncated_dataset__()

	def __build_truncated_dataset__(self):

		mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

		data = mnist_dataobj.data
		target = mnist_dataobj.targets

		if self.dataidxs is not None:
			data = data[self.dataidxs]
			target = target[self.dataidxs]

		return data, target

	def __getitem__(self, index):
		"""
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
		img, target = self.data[index], self.target[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img.numpy(), mode='L')

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)


class CIFAR10_truncated(data.Dataset):

	def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

		self.root = root
		self.dataidxs = dataidxs
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		self.download = download

		self.data, self.target = self.__build_truncated_dataset__()

	def __build_truncated_dataset__(self):

		cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

		data = np.array(cifar_dataobj.data)
		target = np.array(cifar_dataobj.targets)

		if self.dataidxs is not None:
			data = data[self.dataidxs]
			target = target[self.dataidxs]

		return data, target

	def __getitem__(self, index):
		"""
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
		img, target = self.data[index], self.target[index]

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)