import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

file = open("train.txt", "r")
temp = file.readlines()
train = []
for i in range(len(temp)):
	train.append(temp[i].split())

class facesDataset(Dataset):

	def __init__(self, train_data, img_path, transform=None):		
		self.train_data = train_data
		self.img_path = img_path
		self.transform = transform

	def __getitem(self, index):
		img1_path = self.train_data[index][0]
		img2_path = self.train_data[index][1]
		img_label = torch.from_numpy(self.train_data[index][2])
		img1 = Image.open(img1_path)
		img1 = img1.convert('RGB')
		img2 = Image.open(img2_path)
		img2 = img2.convert('RGB')
