# -*- coding: utf-8 -*-
import random
from structure_mask import generate_structure_index, generate_hotmap
import numpy as np
import geatpy as ea
from geatpy import crtpc
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import argparse
import time
import cv2
import os
from Net import Net, VGGNet_Transfer

la = LabelEncoder()
la.fit_transform(['benign', 'malware'])
cate = la.classes_
use_cuda = True
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

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
parser.add_argument('--input_dir', type=str, default='/home/ubuntu/zhan/code/zhan/img-malware/mask-attack/val', help="Input directory with images")
parser.add_argument('--output_dir', type=str, default='./advdata/amg/dense', help="Output directory with images")
parser.add_argument('--mask_dir', type=str, default='./mask', help="Mask directory with images")
#parser.add_argument('--checkpoint_path', type=str, default='/home/zh/adversarial_samples/code/zhan/img-malware/classifier/cnn/models/model_malimg_gray.pth', help=" Path to checkpoint for inception network")
parser.add_argument('--checkpoint_path', type=str, default='/home/ubuntu/zhan/code/zhan/img-malware/mask-attack/models/2-classes/cnn.pth', help=" Path to checkpoint for network")
parser.add_argument('--Diversity', type=bool, default=0, help="use input diversity")
parser.add_argument('--Rotate', type=bool, default=0, help="use input rotate")
parser.add_argument('--method', type=str, default='dilate', help="dilate or erode")
parser.add_argument('--all_slack', type=bool, default=0, help="use all slack part sa mask (suciu method)")
parser.add_argument('--mode', type=str, default='gradcam', help="use gradcam++ or random or idx mode")
args = parser.parse_args()

target_net = 'cnn'

net = torch.load(args.checkpoint_path)
net = net.to(device)
net.eval()

#net = models.resnet50(pretrained=True)
#net.fc = nn.Linear(net.fc.in_features, 2)

#net = models.densenet121(pretrained=False)
#net.classifier = nn.Linear(1024, 2)

#net = models.squeezenet1_1(pretrained=True)
#net.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))

decay = 1
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=3),
     transforms.Resize(size=(224, 224)),
     transforms.ToTensor(),
     ])

num_file = 0
num_success = 0
per_rate = 0
alltime = 0
start_time = time.time()
#image = Image.open('./img.png')
for filename in os.listdir(args.input_dir):

    print(filename)
    num_file = num_file + 1
    image = Image.open(args.input_dir + "/" + filename)

    image_transformed = transform(image)
    image_transformed = transform(image)
    ori_image = image_transformed
    image_transformed = image_transformed.unsqueeze(0)
    image_transformed = image_transformed.to(device)
    output = net(image_transformed)
    original_value, original_idx = torch.max(output, 1)
    print('Predicted:', original_idx.cpu().numpy(), '-', cate[original_idx], '-', original_value.cpu().detach().numpy())
    struct_mask, padding_image = generate_structure_index(np.asarray(image), padding_amount=0, shifting_amount=0, slack_attack=True)
    struct_mask_re = cv2.resize(struct_mask.astype(dtype='float32'), dsize=(args.image_width, args.image_height),
                                interpolation=cv2.INTER_LINEAR)
    _, struct_mask_re = cv2.threshold(struct_mask_re, 0.3, 1, cv2.THRESH_BINARY)
    idx_count = int(sum(struct_mask_re.flatten()))

    mask_idx = np.where(struct_mask_re.flatten() == 1)[0]
    select = []
    select_index = []
    select = np.arange(int(idx_count))
    select_mask = struct_mask_re
    select_index = mask_idx

    heatmap_pp = generate_hotmap(net, padding_image, transform, target_net)

    files_mask = np.zeros([1, args.image_channel, args.image_height, args.image_width])
    files_mask[0, :, :, :] = select_mask
    files_mask = torch.from_numpy(files_mask)
    files_mask = files_mask.to(device)

    im_transformed = transform(Image.fromarray(padding_image))
    im_transformed = im_transformed.unsqueeze(0)
    im_transformed = im_transformed.to(device)

    per_image = torch.Tensor(im_transformed.cpu().data)
    g = 0
    pert_out = im_transformed
    count = 0
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
        pert_out = pert_out + torch.mul(args.alpha * torch.sign(g), files_mask)
        pert_out = torch.clamp(pert_out, 0, 1)
        pred = net(pert_out.type(torch.FloatTensor).cuda())
        predict_value, predict_idx = torch.max(pred, 1)
        per_image = torch.Tensor(pert_out.data.type(torch.FloatTensor))

    if predict_idx != original_idx:
        num_success = num_success + 1
        print('attack success, now erode the mask')
    else:
        continue


    select_mask = np.zeros([3, 224, 224])

    def attackissuccess(ori_image, pert_out, Phen_img):
        files_mask = np.zeros([1, 3, 224, 224])
        files_mask[0, :, :, :] = Phen_img
        files_mask = torch.from_numpy(files_mask)
        ab_files_mask = np.ones([1, 3, 224, 224])
        ab_files_mask[0, :, :, :] = ab_files_mask[0, :, :, :] - Phen_img
        ab_files_mask = torch.from_numpy(ab_files_mask)
        er_pert_out = torch.mul(ori_image.to(device), ab_files_mask.cuda()) + torch.mul(pert_out, files_mask.cuda())
        fipred = net(er_pert_out.type(torch.FloatTensor).cuda())
        predict_value, er_predict_idx = torch.max(fipred, 1)
        result = np.array(er_predict_idx.cpu().data)
        return result

    def aim(Phen, CV, heatmap):
        #Phen_img = np.reshape(Phen, (Nind, 224, 224))
        Phen_img = np.zeros((Nind, 224, 224))
        heat = np.zeros((Nind, 1))
        Pre = np.zeros((Nind, 1))
        num = np.zeros((Nind, 1))
        for i in range(Nind):
            for chrom, idx in enumerate(mask_idx, 0):
                (x, y) = [np.floor(idx / 224).astype(dtype='int64'), np.remainder(idx, 224)]
                Phen_img[i, x, y] = Phen[i, chrom]
            heat[i] = np.sum((heatmap[0][0].cpu().data * Phen_img[i]).numpy()) / np.sum(Phen[i])
            Pre[i] = attackissuccess(ori_image, pert_out, Phen_img[i])
            num[i] = np.sum(Phen[i])

        f = num
        CV = Pre
        return f, CV, heat

    """============================变量设置============================"""
    F1 = np.ones(idx_count)
    F2 = np.zeros(idx_count)
    FieldD = np.array([F1, # 各决策变量编码后所占二进制位数，此时染色体长度为3+2=5
                       F2, # 各决策变量的范围下界
                       F1, # 各决策变量的范围上界
                       F2, # 各决策变量采用什么编码方式(0为二进制编码，1为格雷编码)
                       F2, # 各决策变量是否采用对数刻度(0为采用算术刻度)
                       F1, # 各决策变量的范围是否包含下界(对bs2int实际无效，详见help(bs2int))
                       F1, # 各决策变量的范围是否包含上界(对bs2int实际无效)
                       F1])# 表示两个决策变量都是连续型变量（0为连续1为离散）
    """==========================染色体编码设置========================="""
    #定义种群个数
    Nind = 40
    Encoding = 'BG'
    MAXGEN    = 100; # 最大遗传代数
    maxormins = [1] # 列表元素为1则表示对应的目标函数是最小化，元素为-1则表示对应的目标函数是最大化
    maxormins = np.array(maxormins) # 转化为Numpy array行向量
    selectStyle = 'rws' # 采用轮盘赌选择
    recStyle  = 'xovdp' # 采用两点交叉
    mutStyle  = 'mutbin' # 采用二进制染色体的变异算子
    Lind = int(np.sum(FieldD[0, :])) # 计算染色体长度
    pc        = 0.5 # 交叉概率
    pm        = 1/(Lind+1) # 变异概率
    obj_trace = np.zeros((MAXGEN, 2)) # 定义目标函数值记录器
    var_trace = np.zeros((MAXGEN, Lind)) # 染色体记录器，记录历代最优个体的染色体
    """=========================开始遗传算法进化========================"""
    #Phen =crtpc(Encoding, Nind, FieldD)
    Phen = np.zeros((Nind, idx_count))
    rate = 0.5
    for row in range(Nind):
        samplelist = [i for i in range(idx_count)]
        #list = random.sample(samplelist, idx_count)
        list = random.sample(samplelist, random.randint(int(idx_count*rate*0.2), int(idx_count*rate)))
        for i in list:
            Phen[row, i] = 1
    CV = np.zeros((Nind, 1))
    ObjV, CV, heat = aim(Phen, CV, heatmap_pp)
    FitnV = ea.ranking(ObjV, CV, maxormins) # 根据目标函数大小分配适应度值6f
    best_ind = np.argmax(FitnV) # 计算当代最优个体的序号
    #print(ObjV)
    for gen in range(MAXGEN):
        SelCh = Phen[ea.selecting(selectStyle,FitnV,Nind-1),:] # 选择
        SelCh = ea.recombin(recStyle, SelCh, pc) # 重组
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm) # 变异
        # 把父代精英个体与子代的染色体进行合并，得到新一代种群
        Phen = np.vstack([Phen[best_ind, :], SelCh])
        ObjV, CV, heat = aim(Phen, CV, heatmap_pp) # 求种群个体的目标函数值

        weight = 1/(1+gen*decay)
        ObjV_heat = ObjV + heat*idx_count

        FitnV = ea.ranking(ObjV, CV, maxormins) # 根据目标函数大小分配适应度值
        # 记录
        best_ind = np.argmax(FitnV) # 计算当代最优个体的序号
        obj_trace[gen, 0]=np.sum(ObjV)/ObjV.shape[0] #记录当代种群的目标函数均值
        obj_trace[gen, 1]=ObjV[best_ind] #记录当代种群最优个体目标函数值
        var_trace[gen, :]=Phen[best_ind,:] #记录当代种群最优个体的染色体

    # 进化完成

    #ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']]) # 绘制图像
    """============================输出结果============================"""
    best_gen = np.argmin(obj_trace[:, [1]])
    print('最优解的目标函数值：', obj_trace[best_gen, 1])
    per_r = obj_trace[best_gen, 1] / (224 * 224)
    per_rate = per_rate + per_r

    torch.cuda.empty_cache()
end_time = time.time()
print('success rate:', num_success, '/', num_file, '- avg per-rate', per_rate/num_success, 'time-', (end_time - start_time)/num_file)