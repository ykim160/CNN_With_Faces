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
import sys

# read the training data and split each line
file = open("train.txt", "r")
temp = file.readlines()
train = []
for i in range(len(temp)):
	train.append(temp[i].split())

file = open("test.txt", "r")
temp2 = file.readlines()
test = []
for j in range(len(temp2)):
	test.append(temp2[j].split())

def augmentations(image):
	# Mirror flipping only left to right
	if random.random() < 0.5:
		image = image.transpose(Image.FLIP_LEFT_RIGHT)

	# Rotation between -30 to 30 degrees with respect to the center
	if random.random() < 0.5:
		rand_angle = random.randint(-30, 30)
		image = image.rotate(rand_angle)

	# Translation within -10 and +10 pixels
	if random.random() < 0.5:
		move_x = random.randint(-10, 10)
		move_y = random.randint(-10, 10)
		image = image.transform(image.size, Image.AFFINE, (1,0,move_x,0,1,move_y))	

	# Scaling within 0.7 to 1.3
	if random.random() < 0.5:
		rand_ratio = 0.6 * random.random() + 0.7
		og_size = image.size
		new_size = tuple([int(i*rand_ratio) for i in image.size])
		image = image.resize(new_size, Image.ANTIALIAS)
		if rand_ratio <= 1:
			image = image.crop((0,0,og_size[0],og_size[1]))
		else:
			left = abs((og_size[0] - new_size[0])/2)
			right = abs((og_size[0] + new_size[0])/2)
			top = abs((og_size[1] - new_size[1])/2)
			bottom = abs((og_size[1] + new_size[1])/2)
			image = image.crop((left,top,right,bottom))

	return image

class facesDataset(Dataset):

	def __init__(self, train_data, transform=None, augment=False):
		self.train_data = train_data
		self.transform = transform
		self.augment = augment

	def __getitem__(self, index):
		img1_path = 'lfw/' + self.train_data[index][0]
		img2_path = 'lfw/' + self.train_data[index][1]
		img_label = map(float,self.train_data[index][2])
		img_label = torch.from_numpy(np.array(img_label)).float()
		img1 = Image.open(img1_path).convert('RGB')
		img2 = Image.open(img2_path).convert('RGB')

		# if we need to augment it goes through this process
		prob_augment = (random.random() <= 0.7)
		# it augments with a probability of 70%
		if self.augment == True and prob_augment == True:
			img1 = augmentations(img1)
			img2 = augmentations(img2)

		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)

		return img1, img2, img_label
		
	def __len__(self):
		return len(self.train_data)

class SiameseNetwork(nn.Module):
	def __init__(self):
		super(SiameseNetwork, self).__init__()
		self.cnn1 = nn.Sequential(
			nn.Conv2d(3,64,5,padding=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(2,stride=2),
			nn.Conv2d(64,128,5,padding=2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(128),
			nn.MaxPool2d(2,stride=2),
			nn.Conv2d(128,256,3,padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(2,stride=2),
			nn.Conv2d(256,512,3,padding=1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(512),
		)

		self.fc1 = nn.Sequential(
			nn.Linear(131072,1024),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(1024),
		)

		self.fc2 = nn.Sequential(
			nn.Linear(2048, 1),
			nn.Sigmoid()
		)

	def forward_once(self, x):
		tmp = self.cnn1(x)
		tmp = tmp.view(tmp.size()[0], -1)
		result = self.fc1(tmp)
		return result

	def forward(self, input1, input2):
		f1 = self.forward_once(input1)
		f2 = self.forward_once(input2)
		f12 = torch.cat((f1, f2),1)
		result = self.fc2(f12)
		return result

if '--save' in sys.argv:
	trans = transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()])
	train_set = facesDataset(train_data=train, transform=trans, augment=True)
	train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2)

	net = SiameseNetwork().cuda()
	function = nn.BCELoss()
	learning_rate = 1e-6
	optimizer = optim.Adam(net.parameters(),lr = learning_rate)
	epochs = 10

	for epoch in range(epochs):
		for i, data in enumerate(train_loader,0):
			img0, img1, label = data
			img0 = Variable(img0).cuda()
			img1 = Variable(img1).cuda()
			label = Variable(label).cuda()
			optimizer.zero_grad()
			y_pred = net(img0, img1)
			loss = function(y_pred, label)
			loss.backward()
			optimizer.step()

			if i % 10 == 0:
				print("Epoch %d, Batch %d, Loss %f" % (epoch, i ,loss.data[0]))

	weight_index = sys.argv.index('--save') + 1
	weight_path = sys.argv[weight_index]
	torch.save(net.state_dict(), weight_path)

elif '--load' in sys.argv:
	weight_index = sys.argv.index('--load') + 1
	weight_path = sys.argv[weight_index]

	net = SiameseNetwork().cuda()
	net.load_state_dict(torch.load(weight_path))
	net.eval()
	function = nn.BCELoss()

	# Testing on training data set
	trans1 = transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()])
	train_set1 = facesDataset(train_data=train, transform=trans1)
	test_loader1 = DataLoader(train_set1, batch_size=8, shuffle=False)
	cum_correct = 0

	for i, data in enumerate(test_loader1):
		img0, img1, label = data
		label = label.view(label.numel(),-1)
		img0 = Variable(img0, volatile=True).cuda()
		img1 = Variable(img1, volatile=True).cuda()
		label = Variable(label, volatile=True).cuda()
		y_pred = net(img0, img1)
		loss = function(y_pred, label)
		y_round = torch.round(y_pred)
		if i % 10 == 0:
			print ("Batch %d, Loss %f" % (i, loss.data[0]))
		cum_correct += (y_round.view(-1) == label.view(-1)).sum().float()

	avg_correct = cum_correct / float(len(train_set1))

	print "Training set prediction Accuracy is: ", avg_correct * 100

	# Testing on testing data set
	trans2 = transforms.Compose([transforms.Scale((128,128)),transforms.ToTensor()])
	train_set2 = facesDataset(train_data=test, transform=trans2)
	test_loader1 = DataLoader(train_set2, batch_size=8, shuffle=False)
	cum_correct = 0

	for i, data in enumerate(test_loader1):
		img0, img1, label = data
		label = label.view(label.numel(),-1)
		img0 = Variable(img0, volatile=True).cuda()
		img1 = Variable(img1, volatile=True).cuda()
		label = Variable(label, volatile=True).cuda()
		y_pred = net(img0, img1)
		loss = function(y_pred, label)
		y_round = torch.round(y_pred)
		if i % 10 == 0:
			print ("Batch %d, Loss %f" % (i, loss.data[0]))
		cum_correct += (y_round.view(-1) == label.view(-1)).sum().float()

	avg_correct = cum_correct / float(len(train_set2))

	print "Testing set prediction Accuracy is: ", avg_correct * 100
else:
	print "Please use either --save MODEL_NAME or --load MODEL_NAME"
