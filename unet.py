import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
import torch.tensor
from torchvision import datasets, transforms
from torch.autograd import Variable

# Add your own dataset as data_loader

# Hyperparameters
epochs = 100
lr = 0.0001


def conv2x2(in_c, out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
	if useBN:
		return nn.Sequential(
			      nn.Conv2d(in_c, out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
			      nn.BatchNorm2d(out),
			      nn.LeakyReLU(0.2),
			      nn.Conv2d(out, out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
			      nn.BatchNorm2d(out),
			      nn.LeakyReLU(0.2))

	else:
		return nn.Sequential(
					nn.Conv2d(in_c, out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
					nn.ReLU(),
					nn.Conv2d(out, out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
					nn.ReLU()
					)

def concat(c, f, in_c, in_f, up):
	conv1 = nn.ConvTranspose2d(c, f, 4, 2, 1, bias=False)
	torch.cat(conv1, f)

	return nn.Sequential(nn.ConvTranspose2d(c, f, 4, 2, 1, bias=False))
	upsample(c)

def upsample(c, f):
	return nn.Sequential(
		nn.ConvTranspose2d(c, f, 4, 2, 1, bias=False),
		nn.ReLU())


class UNet(nn.Module):
	def __init__(self, useBN=False):
		super(UNet, self).__init__()

		self.conv1   = conv2x2(1, 32, useBN=useBN)
		self.conv2   = conv2x2(32, 64, useBN=useBN)
		self.conv3   = conv2x2(64, 128, useBN=useBN)
		self.conv4   = conv2x2(128, 256, useBN=useBN)
		self.conv5   = conv2x2(256, 512, useBN=useBN)

		self.conv4m = conv2x2(512, 256, useBN=useBN)
		self.conv3m = conv2x2(256, 128, useBN=useBN)
		self.conv2m = conv2x2(128, 64, useBN=useBN)
		self.conv1m = conv2x2(64, 32, useBN=useBN)

		self.conv0  = nn.Sequential(
        nn.Conv2d(32, 1, 3, 1, 1),
        nn.Sigmoid())

		self.max_pool = nn.MaxPool2d(2)

		self.upsample54 = upsample(512, 256)
		self.upsample43 = upsample(256, 128)
		self.upsample32 = upsample(128, 64)
		self.upsample21 = upsample(64, 32)

		## weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, x):
		output1 = self.max_pool(self.conv1(x))
		output2 = self.max_pool(self.conv2(output1))
		output3 = self.max_pool(self.conv3(output2))
		output4 = self.max_pool(self.conv4(output3))
		output5 = self.max_pool(self.conv5(output4))

		conv5m_out = torch.cat((self.upsample54(output5), output4), 1)
		conv4m_out = self.conv4m(conv5m_out)

		conv4m_out = torch.cat((self.upsample54(output4), output3), 1)
		conv3m_out = self.conv3m(conv4m_out)

		conv3m_out = torch.cat((self.upsample54(output3), output2), 1)
		conv2m_out = self.conv2m(conv3m_out)

		conv2m_out = torch.cat((self.upsample54(output2), output1), 1)
		conv1m_out = self.conv1m(conv2m_out)

		final = self.conv0(conv1m_out)
		return final


model = UNet(True)

# Optimizer and Mean-Squared Loss
optimizer = optim.Adagrad(model.parameters(), lr=lr)
criterion = nn.MSELoss()
loss_sum = 0

# Training:
for i, (x, y) in enumerate(data_loader):
	x = Variable(x)
	ground_truth = Variable(y)

	for a in range(epochs):
		prediction = model(x)
		loss = criterion(prediction, ground_truth)

		optimizer.zero_grad()
		loss.backward()
		loss_sum += loss.data[0]
		optimizer.step()

		print("epoch ", a, "loss ", loss.data[0])










