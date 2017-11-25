import torch
import cv2
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

def augmentations(image):
	# Mirror flipping
	#if random.random() < 0.5:
		image = image.transpose(Image.FLIP_LEFT_RIGHT)

temp_img = Image.open('lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg').convert('RGB')
temp_img = augmentations(temp_img)
cv2.imwrite('test.jpg', temp_img)
'''
class facesDataset(Dataset):

	def __init__(self, train_data, img_path, transform=None, augment=False):
		self.train_data = train_data
		self.img_path = img_path
		self.transform = transform

	def __getitem__(self, index):
		img1_path = 'lfw/' + self.train_data[index][0]
		img2_path = 'lfw/' + self.train_data[index][1]
		img_label = self.train_data[index][2]
		img1 = Image.open(img1_path).convert('RGB')
		img2 = Image.open(img2_path).convert('RGB')

		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)

		# if we need to augment it goes through this process
		prob_augment = (random.random() <= 0.7)
		# it augments with a probability of 70%
		if self.augment == True and prob_agument == True:
					
		

	def __len__(self);
		return len(self.train_data)
'''
