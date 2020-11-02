import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
    	self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
           # print(name,module)
           # print('*'*20)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs,x

class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """
	def __init__(self, model, target_layers):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations,output= self.feature_extractor(x)
	#	print(output.size())
#		output = output.view(output.size(0), -1)
		return target_activations,output

def preprocess_image(img):
	means=[0.5, 0.5, 0.5]
	stds=[0.5, 0.5, 0.5]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask,root):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	both=np.concatenate((img,cam),axis=1)
	cv2.imwrite(root, np.uint8(255 * both))

class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, target_layer_names)

	def forward(self, input):
		return self.model(input) 

	def __call__(self, input, index = None):
		if self.cuda:
			_,att=self.forward(input.cuda())
			features,output = self.extractor(input.cuda())
		else:
			_,att=self.forward(input)
			features,output = self.extractor(input)
                
	#	ind = torch.min(output, 1)[1].cpu()  # 取每一个图像中判别最不真的块的>索引
	#	index = torch.cat((torch.arange(1).long(), ind), 0).view(2, -1)
	#	index = index.numpy().tolist()
	#	layer = att[index]
	#	index = torch.topk(layer, 5)[1].cpu().numpy()
	#	print(output.size())
		output=output[:,:,1:6,1:6].contiguous().view(1,-1)
	#	print(output.size())
		if index == None:
                   index = torch.topk(output,10,largest=False)[1].cpu().numpy()
		five_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		five_hot[0][index] = 1
	#	print(five_hot)
		one_hot = Variable(torch.from_numpy(five_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()
	#	print(one_hot)
		one_hot.backward(retain_graph=True)
	#	print(self.extractor.get_gradients())
		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

		target = features[-1]
		target = target.cpu().data.numpy()[0, :]
	#	print(target.shape)
		weights = np.mean(grads_val, axis = (2, 3))[0, :]
	#	weights=np.mean(grads_val,axis=(0,1))[0,:]
	#	print(grads_val.shape)
		cam = np.zeros(target.shape[1 : ], dtype = np.float32)

		for i,w in enumerate(weights):
			cam += w*target[i, :, :]
	#	cam=cam*weights

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (128, 128))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		# replace ReLU with GuidedBackpropReLU
		for idx, module in self.model._modules.items():
			if module.__class__.__name__ == 'LeakyReLU':
				self.model.features._modules[idx] = GuidedBackpropReLU()

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index = None):
		if self.cuda:
			output,_ = self.forward(input.cuda())
		else:
			output,_ = self.forward(input)

		if index == None:
			index= torch.topk(output, 5)[1].cpu().numpy()

		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		# self.model.features.zero_grad()
		# self.model.classifier.zero_grad()
		one_hot.backward(retain_graph=True)

		output = input.grad.cpu().data.numpy()
		output = output[0,:,:,:]

		return output

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='./examples/both.png',
	                    help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")

	return args


