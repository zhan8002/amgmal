# -*- coding: utf-8 -*-
import cv2
import struct
import math
import os
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
from util import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp
import torch.nn as nn

def generate_structure_index(img, padding_amount = 0, shifting_amount = 0, slack_attack =True):
    #img_path =  '/home/zh/adversarial_samples/dataset/Malimg/dataset_9010/dataset_9010/malimg_dataset/validation/Dontovo.A/04c497337c81115d7d5df32ebc6458d1.png'
    #with open(img_path, "rb") as f:
    #    bin_contents = f.read()
    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    im_height, im_width =img.shape
    arrary_bytes = img.tobytes()
    code = bytearray(arrary_bytes)

    preferable_extension_amount = shifting_amount
    #padding_amount = 512

    pe_position = code[60]
    optional_header_size = code[pe_position + 20]
    coff_header_size = 24
    section_entry_length = 40
    size_of_raw_data_pointer = 20

    section_file_alignment = struct.unpack("<I", code[pe_position + coff_header_size + 36: pe_position + coff_header_size + 40])[0]

    content_offset = pe_position+optional_header_size+coff_header_size + 12
    first_content_offset = struct.unpack("<I", code[content_offset:content_offset + 4])[0]
    extension_amount = int(math.floor(preferable_extension_amount / section_file_alignment)) * section_file_alignment

    x_code = code

    #shift part
    index_to_shift = list(range(first_content_offset, first_content_offset + extension_amount))
    for i in range(code[pe_position + 6]):
        shift_position = (
                pe_position
                + coff_header_size
                + optional_header_size
                + i * section_entry_length
                + size_of_raw_data_pointer
        )
        old_value = struct.unpack("<I", x_code[shift_position:shift_position + 4])[0]
        new_value = old_value + extension_amount
        new_value = struct.pack("<I", new_value)
        x_code[shift_position:shift_position + 4] = new_value
    x_code = x_code[:first_content_offset] + b'\x00' * extension_amount + x_code[first_content_offset:]

    #Slack part
    if slack_attack == True:
        index_to_slack = []
        for i in range(code[pe_position + 6]):
            virtual_size_position = (
                    pe_position
                    + coff_header_size
                    + optional_header_size
                    + i * section_entry_length
                    + 8
            )
            virtual_size = struct.unpack("<I", code[virtual_size_position:virtual_size_position+4])[0]
            raw_size = struct.unpack("<I", code[virtual_size_position + 8:virtual_size_position+12])[0]
            offset = struct.unpack("<I", code[virtual_size_position+12:virtual_size_position+12+4])[0]

            if virtual_size < raw_size:
                index_to_slack.extend(list(range(offset + virtual_size, offset + raw_size)))

    #Padding part

    index_to_padding = list(range(len(x_code), len(x_code) + padding_amount))
    x_code = x_code[:len(x_code)] + b'\x00' * padding_amount

    #合并，保持原始图像的宽度
    mask_list = index_to_shift + index_to_slack + index_to_padding
    mask = np.zeros(len(x_code), dtype=np.int8)


    for index in mask_list:
        if index < len(mask.flatten()):
            mask[index] = 1
    #for i in range(im_width - np.remainder(len(x_code), im_width)):
    #    mask = np.append(mask, 1)

    #x_code = x_code[:len(x_code)] + b'\x00' * (im_width - np.remainder(len(x_code), im_width))
    height = np.ceil(len(x_code)/im_width).astype(int)

    mask = np.reshape(mask, (height, im_width))

    padding_code = np.reshape(np.asarray(x_code), (im_height, im_width))
    #print("generate structure mask completed, the number of index is", len(mask_list))
    return mask, padding_code



def generate_hotmap(net, image, transform, target_net):


    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    normed_torch_img = torch_img.unsqueeze(0)
    normed_torch_img = normed_torch_img.cuda()


    cam_dict = dict()

    if target_net == 'cnn':
        squeezenet = torch.load('/home/ubuntu/zhan/code/zhan/img-malware/mask-attack/models/2-classes/squeeze.pth')
        #vgg = net
        squeezenet.eval()
        squeezenet.cuda()

        squeezenet_model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation',
                                     input_size=(224, 224))
        #vgg_gradcam = GradCAM(vgg_model_dict, True)
        gradcampp = GradCAMpp(squeezenet_model_dict, True)

    if target_net == 'vgg':
        #vgg = torch.load('./models/2-classes/vgg_2cls.pth')
        vgg = net
        vgg.eval()
        vgg.cuda()

        vgg_model_dict = dict(type='vgg', arch=vgg, layer_name='features_29', input_size=(224, 224))
        #vgg_gradcam = GradCAM(vgg_model_dict, True)
        gradcampp = GradCAMpp(vgg_model_dict, True)


    if target_net == 'res':
        resnet = net
        resnet.eval()
        resnet.cuda()

        resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        #resnet_gradcam = GradCAM(resnet_model_dict, True)
        gradcampp = GradCAMpp(resnet_model_dict, True)


    if target_net == 'dense':
        densenet = net
        densenet.eval()
        densenet.cuda()

        densenet_model_dict = dict(type='densenet', arch=densenet, layer_name='features_norm5', input_size=(224, 224))
        #densenet_gradcam = GradCAM(densenet_model_dict, True)
        gradcampp = GradCAMpp(densenet_model_dict, True)


    if target_net == 'squeeze':
        squeezenet = net
        squeezenet.eval()
        squeezenet.cuda()

        squeezenet_model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation',
                                     input_size=(224, 224))
        #squeezenet_gradcam = GradCAM(squeezenet_model_dict, True)
        gradcampp = GradCAMpp(squeezenet_model_dict, True)
    gradcam_pp, _ = gradcampp(normed_torch_img)

    return gradcam_pp[0]

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(53*53*16, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 25),
        )