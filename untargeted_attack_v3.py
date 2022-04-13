#!/bin/env python
import numpy as np
import argparse
import os
import random
import cv2
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from structure_mask import generate_structure_index, generate_hotmap
from util import bisectSearch
import time

la = LabelEncoder()
la.fit_transform(['benign', 'malware'])
cate = la.classes_

# 定义超参数
parser = argparse.ArgumentParser()
parser.add_argument('--image_width', type=int, default=224, help="Width of each input images")
parser.add_argument('--image_height', type=int, default=224, help="Height of each input images")
parser.add_argument('--image_resize', type=int, default=224, help="Resize scale")
parser.add_argument('--image_channel', type=int, default=3, help="Gray or RGB.")
parser.add_argument('--iter_num', type=int, default=250, help="iteration numbers")
parser.add_argument('--prob', type=float, default=0.7, help="probability of using diverse inputs")
parser.add_argument('--decay_factor', type=float, default=1.0, help="momentum weight")
parser.add_argument('--alpha', type=float, default=0.3, help="iteration parameter")
parser.add_argument('--input_dir', type=str, default='./val', help="Input directory with images")
parser.add_argument('--checkpoint_path', type=str, default='./models/2-classes/squeeze.pth', help=" Path to checkpoint for network")
parser.add_argument('--method', type=str, default='dilate', help="dilate or erode")
parser.add_argument('--all_slack', type=bool, default=0, help="use all slack part sa mask (suciu method)")
parser.add_argument('--mode', type=str, default='gradcam', help="use gradcam++ or random or idx mode")
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

#定义模型和超参数
args.method = 'dilate'
args.mode = 'gradcam'
args.all_slack = 1
target_net = 'squeeze'

# Load model
net = torch.load(args.checkpoint_path)
net = net.to(device)
net.eval()

###############################################################
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=args.image_channel),
     transforms.Resize(size=(args.image_width, args.image_height)),
     transforms.ToTensor(),
     ])

num_file = 0
num_success = 0
per_rate = 0

start = time.time()
for filename in os.listdir(args.input_dir):
    print(filename)
    image = Image.open(args.input_dir + "/" + filename)
    image_transformed = transform(image)

    # Get the initial predictions
    image_transformed = image_transformed.unsqueeze(0)
    image_transformed = image_transformed.to(device)
    output = net(image_transformed)
    original_value, original_idx = torch.max(output, 1)
    print('Predicted:', original_idx.cpu().numpy(), '-', cate[original_idx], '-', original_value.cpu().detach().numpy())

    # Get the index of Slack
    struct_mask, padding_image = generate_structure_index(np.asarray(image), slack_attack=True)

    # Get the hotmap
    heatmap_pp = generate_hotmap(net, padding_image, transform, target_net)

    struct_mask_re = cv2.resize(struct_mask.astype(dtype='float32'), dsize=(args.image_width, args.image_height), interpolation=cv2.INTER_LINEAR)
    _, struct_mask_re = cv2.threshold(struct_mask_re, 0.3, 1, cv2.THRESH_BINARY)
    heatmask = (heatmap_pp[0][0].cpu().data * struct_mask_re).numpy()
    idx_count = sum(struct_mask_re.flatten())

    ##使用轮盘选择法挑选热力值高的像素
    mask_idx = np.where(struct_mask_re.flatten() == 1)[0]
    fitness = []
    select = []
    select_index = []
    for f_idx in mask_idx:
        fitness.append((heatmask.flatten())[f_idx]+0.1)

    # dilate method 初始不选择点
    if args.method == 'dilate':
        if args.all_slack == 1:
            ori_rate = 1
        else:
            ori_rate = 0
        mask_count = int(idx_count*ori_rate)
        i = 0
        while i < mask_count and mask_count > 0:
            if args.mode == 'random':
                cho = random.randint(0, mask_count - 1)
            elif args.mode == 'idx':
                cho = i
            else:
                cho = bisectSearch(fitness)
                fitness[cho] = 0

            if cho not in select:
                select.append(cho)
                select_index.append(mask_idx[cho])
                i = i + 1
        select_mask = np.zeros(shape=heatmask.shape)

        for idx in select_index:
            (x, y) = [np.floor(idx / heatmask.shape[0]).astype(dtype='int64'), np.remainder(idx, heatmask.shape[0])]
            select_mask[x, y] = 1

    # erode method 初始选择所有的点
    if args.method == 'erode':
        select = np.arange(int(idx_count))
        select_mask = struct_mask_re
        select_index = mask_idx

    print("the original number of insert is", len(select_index), '/', int(idx_count))

    files_mask = np.zeros([1, args.image_channel, args.image_height, args.image_width])
    files_mask[0, :, :, :] = select_mask

    im_transformed = transform(Image.fromarray(padding_image))
    im_transformed = im_transformed.unsqueeze(0)
    im_transformed = im_transformed.to(device)

    count = 0
    pert_out = image_transformed
    g = 0
    files_mask = torch.from_numpy(files_mask)
    predict_idx = original_idx

    if args.method == 'dilate':
        per_image = torch.Tensor(im_transformed.cpu().data)
        while count < args.iter_num and original_idx == predict_idx:
            count += 1
            # Optimize the patch
            per_image = per_image.to(device)
            per_image.requires_grad = True
            output = net(per_image)
            loss = F.nll_loss(output, original_idx)
            net.zero_grad()
            if per_image.grad is not None:
                per_image.grad.data.fill_(0)
            loss.backward(retain_graph=True)
            grad = per_image.grad.data

            g = args.decay_factor * g + grad / torch.norm(grad, p=1)
            pert_out = pert_out + torch.mul(args.alpha * torch.sign(g), files_mask.cuda())
            pert_out = torch.clamp(pert_out, 0, 1)

            pred = net(pert_out.type(torch.FloatTensor).cuda())
            predict_value, predict_idx = torch.max(pred, 1)

            if count % (args.iter_num/10) == 0:
                print('num-', len(select), '-predict:', cate[predict_idx], 'with confidence ', predict_value.cpu().detach().numpy())

            if (count-1) % 10 == 0 and count <= 100:
            # dilate mask 扩大mask的范围
                if args.method == 'dilate':
                    i = 0
                    while i < int(idx_count*(1-ori_rate)/10):
                        if args.mode == 'random':
                            cho = random.randint(0, idx_count-1)
                        elif args.mode == 'idx':
                            cho = int(i + int(idx_count*(1-ori_rate)/10)*(count-1)/10)
                        else:
                            cho = bisectSearch(fitness)
                            fitness[cho] = 0

                        if cho not in select:
                            select.append(cho)
                            select_index.append(mask_idx[cho])
                            i = i + 1

                    for idx in select_index:
                        (x, y) = [np.floor(idx / heatmask.shape[0]).astype(dtype='int64'), np.remainder(idx, heatmask.shape[0])]
                        select_mask[x, y] = 1
                    files_mask = np.zeros([1, args.image_channel, args.image_height, args.image_width])
                    files_mask[0, :, :, :] = select_mask
                    files_mask = torch.from_numpy(files_mask)
            per_image = torch.Tensor(pert_out.data.type(torch.FloatTensor))
    if args.method == 'erode':
        per_image = torch.Tensor(image_transformed.cpu().data)
        ori_image = per_image
        while count < args.iter_num:
            count += 1
            # Optimize the patch
            per_image = per_image.to(device)
            per_image.requires_grad = True
            output = net(per_image)
            loss = F.nll_loss(output, original_idx)
            net.zero_grad()
            loss.backward()
            grad = per_image.grad.data

            g = args.decay_factor * g + grad / torch.norm(grad, p=1)
            pert_out = pert_out + torch.mul(args.alpha * torch.sign(g), files_mask.cuda())
            pert_out = torch.clamp(pert_out, 0, 1)

            pred = net(pert_out.type(torch.FloatTensor).cuda())
            predict_value, predict_idx = torch.max(pred, 1)

            per_image = torch.Tensor(pert_out.data.type(torch.FloatTensor))
        if args.all_slack == 1:
            print('Final Predicted:', predict_idx.cpu().detach().numpy(), '-', cate[predict_idx], 'iter-', count,
                  'num-', len(select))

            num_file = num_file + 1
            if original_idx != predict_idx:
                num_success = num_success + 1

                per_r = len(select) / (224 * 224)
                per_rate = per_rate + per_r
            continue

        if predict_idx != original_idx:
            print('attack success, now erode the mask')

        er_predict_idx = predict_idx
        erode_num = 0
        while original_idx != er_predict_idx and len(select) > 0:
            er_select = select
            er_select_index = select_index
            select_mask = np.zeros(shape=heatmask.shape)

            i = 0
            while i < int(idx_count / 20) and len(er_select) > 0:
                if args.mode == 'random':
                    cho = er_select[random.randint(0, len(er_select)-1)]
                elif args.mode == 'idx':
                    cho = int(i + int(idx_count/20)*erode_num)
                else:
                    #cho = bisectSearch(list(map(lambda x: 10-x, fitness)))
                    cho = np.argsort(fitness)[0]

                    if cho < len(fitness):
                        fitness[cho] = 10
                if cho in er_select:
                    er_select = np.delete(er_select, np.where(er_select == cho))
                    er_select_index = np.delete(er_select_index, np.where(er_select_index == mask_idx[cho]))
                    i = i + 1
            erode_num = erode_num + 1
            for idx in er_select_index:
                (x, y) = [np.floor(idx / heatmask.shape[0]).astype(dtype='int64'), np.remainder(idx, heatmask.shape[0])]
                select_mask[x, y] = 1

            files_mask = np.zeros([1, args.image_channel, args.image_height, args.image_width])
            files_mask[0, :, :, :] = select_mask
            files_mask = torch.from_numpy(files_mask)
            ab_files_mask = np.ones([1, args.image_channel, args.image_height, args.image_width])
            ab_files_mask[0, :, :, :] = ab_files_mask[0, :, :, :] - select_mask
            ab_files_mask = torch.from_numpy(ab_files_mask)

            er_pert_out = torch.mul(ori_image.to(device), ab_files_mask.cuda()) + torch.mul(pert_out, files_mask.cuda())
            pred = net(er_pert_out.type(torch.FloatTensor).cuda())
            predict_value, er_predict_idx = torch.max(pred, 1)
            if original_idx != er_predict_idx:
                select = er_select
                select_index = er_select_index
                pert_out = er_pert_out
                predict_idx = er_predict_idx
                print( 'num-', len(select),'-predict:', cate[er_predict_idx], 'with confidence ', predict_value.cpu().detach().numpy())

    pert_out = pert_out.cpu().detach().numpy()
    print('Final Predicted:', predict_idx.cpu().detach().numpy(), '-', cate[predict_idx], 'iter-', count, 'num-', len(select))

    num_file = num_file + 1
    if original_idx != predict_idx:
        num_success = num_success + 1

        per_r = len(select)/(args.image_width*args.image_length)
        per_rate = per_rate + per_r
    torch.cuda.empty_cache()
end = time.time()
print('success rate:', num_success, '/', num_file, '- avg per-rate', per_rate/num_success, 'time-', (end-start)/num_file)
