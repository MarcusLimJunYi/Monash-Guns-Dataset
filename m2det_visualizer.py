from __future__ import print_function
import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import shutil
from m2det import build_net
from utils.timer import Timer
import torch.backends.cudnn as cudnn
from layers.functions import Detect,PriorBox
from data import BaseTransform
from configs.CC import Config
from tqdm import tqdm
from utils.core import *
import torchvision.transforms as transforms
import torchvision.models as models
import time
from layers.nn_utils import *
from torch.nn import init as init
from utils.core import print_info
import matplotlib.pyplot as plt

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layer,target_scale):
        self.model = model
        self.target_layer = target_layer
        self.target_scale = target_scale
        self.gradients = []
        self.phase = "train"

    def save_gradient(self, grad):
        # print(grad)
        self.gradients.append(grad)


    def __call__(self, x):
        outputs = []
        self.gradients = []

        loc,conf = list(),list()
        base_feats = list()
        # for k in range(len(self.model.base)):
        #     x = self.model.base[k](x)
        #     if k in [22,34]:
        #         base_feats.append(x)

        #Resnet101 and FFMv3
        base_feats = self.model.base(x, [1,2,4])

        # base_feature = torch.cat(
        #         (self.model.reduce(base_feats[0]), F.interpolate(self.model.up_reduce(base_feats[1]),scale_factor=2,mode='nearest')),1
        #         )

        base_feature = torch.cat(
                (F.interpolate(self.model.reduce_shallow(base_feats[0]),scale_factor=0.5,mode='nearest'),self.model.reduce_medium(base_feats[1]),
                    F.interpolate(self.model.up_reduce(base_feats[2]),scale_factor=2,mode='nearest')),1
                )

        # tum_outs is the multi-level multi-scale feature
        tum_outs = [getattr(self.model, 'unet{}'.format(1))(self.model.leach[0](base_feature), 'none')]
        for i in range(1,8,1):
            tum_outs.append(
                    getattr(self.model, 'unet{}'.format(i+1))(
                        self.model.leach[i](base_feature), tum_outs[i-1][-1]
                        )
                    )

        tum_outs[self.target_layer][self.target_scale].register_hook(self.save_gradient) 
        outputs += [tum_outs[self.target_layer][self.target_scale]]
       
        # concat with same scales
        sources = [torch.cat([_fx[i-1] for _fx in tum_outs],1) for i in range(6, 0, -1)]

        # forward_sfam     
        # sources = self.model.sfam_module(sources)
        sources[0] = self.model.Norm(sources[0])

        for i,(x,l,c) in enumerate(zip(sources, self.model.loc, self.model.conf)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        return outputs, conf

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.model.softmax(conf.view(-1, 2)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, 2),
        )

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layer,target_scale):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layer,target_scale)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        output = output.view(output.size(0), -1, 2)
        output = output[:,:,1]
        return target_activations, output

class GradCam:
    def __init__(self, model, target_layer_names, target_scale_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, target_scale_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        
        one_hot = np.zeros((output.size()[0], output.size()[-1]), dtype = np.float32)
        one_hot[:,index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)

        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        self.model.zero_grad()
        # self.model.conf.zero_grad()
        one_hot.backward(retain_graph=True)

        # print(self.extractor.get_gradients())
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        # print(self.model.unet1)
        # print(features[-1].size())
        # print(output.size())

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        if np.sum(cam) < 0:
            cam=-cam
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))
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
            # print(module._modules.items())
            for sub_idx, sub_module in module._modules.items():
                if sub_module.__class__.__name__ == 'ReLU' :
                    module._modules[sub_idx] = GuidedBackpropReLU()

                for sub_sub_idx, sub_sub_module in sub_module._modules.items():

                    if sub_sub_module.__class__.__name__ == 'ReLU' :
                        sub_module._modules[sub_sub_idx] = GuidedBackpropReLU()

                    if sub_sub_module.__class__.__name__ == 'BasicConv':
                        for sub_sub_sub_idx, sub_sub_sub_module in sub_sub_module._modules.items():
                            if sub_sub_sub_module.__class__.__name__ == 'ReLU' : 
                                sub_sub_module._modules[sub_sub_sub_idx] = GuidedBackpropReLU()
        # print(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        # print(input.grad)
        if self.cuda:
            loc,output = self.forward(input.cuda())
        else:
            loc,output =  self.forward(input)

        output = output[:,1].unsqueeze(0)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.sfam_module.zero_grad()
        # self.model.softmax.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

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

def show_cam_on_image(img, mask,target_layer,target_scale):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    # cv2.imshow('cam',np.uint8(255 * cam))
    cv2.imwrite("grad_cam/cam_{}_{}.jpg".format(target_layer,target_scale), np.uint8(255 * cam))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./Demo_EAAI/COCO_val2014_268.jpg',
                        help='Input image path')
    parser.add_argument('-c', '--config', default='configs/m2det512_resnet101.py', type=str)
    parser.add_argument('-m', '--trained_model', default='weights/Xiaomi_IMDB_Combinedtrain_ANCHOR_Default_GIoU_FL_5_025_Resnet101_FFMv3_Final.pth', type=str, help='Trained state_dict file path to open')
    parser.add_argument('--target_layers', type=int, default=8,help='Target layer')
    parser.add_argument('--target_scales', type=int, default=6,help='Target scale')
    args = parser.parse_args()

    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    global cfg
    cfg = Config.fromfile(args.config)
    anchor_config = anchors(cfg)
    print_info('The Anchor info: \n{}'.format(anchor_config))
    priorbox = PriorBox(anchor_config)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (512, 512))) / 255
    input = preprocess_image(img)

    for target_layer in range(args.target_layers):
        for target_scale in range(args.target_scales):
            print(target_layer)
            print(target_scale)
            net = build_net('test',
                    size = cfg.model.input_size,
                    config = cfg.model.m2det_config) 
            init_net(net, cfg, args.trained_model)

            grad_cam = GradCam(model = net, \
                            target_layer_names = target_layer, 
                            target_scale_names = target_scale,
                            use_cuda=args.use_cuda)


            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested index.
            target_index = None

            mask = grad_cam(input, target_index)

            show_cam_on_image(img, mask,target_layer,target_scale)

            gb_model = GuidedBackpropReLUModel(model = net, use_cuda=args.use_cuda)
            gb = gb_model(input, index=target_index)
            utils.save_image(torch.from_numpy(gb), 'gb.jpg')

            cam_mask = np.zeros(gb.shape)
            for i in range(0, gb.shape[0]):
                cam_mask[i, :, :] = mask

            cam_gb = np.multiply(cam_mask, gb)
            utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')