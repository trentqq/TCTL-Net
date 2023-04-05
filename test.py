# -- coding: utf-8 --
import argparse
import os
import torch
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import cv2
from model.TCTLNet_model import TCTLNet
from data.aligned_dataset import TestDataset


def Test(args):
    input_data = TestDataset(args.dataroot, input_size=args.input_size, output_size=args.output_size)
    input_dataloader = DataLoader(input_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    if args.suffix:
        save_dir = args.name + '_' + args.suffix
    else:
        save_dir = args.name
    save_path = os.path.join(args.results_dir, save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    net = TCTLNet(output_size=args.output_size, init_weights=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.load_state_dict(torch.load(os.path.join(args.model_path, args.name, 'net_latest.pth')))
    net.eval()

    test_bar = tqdm(input_dataloader)

    with torch.no_grad():

        for step, (img_lab_unet, pic_vgg, pic_name, img_lab, gray_L, gray_A, gray_B, mean, std) in enumerate(test_bar):
            name1 = str(list(pic_name)).lstrip("['").rstrip("']")
            save_name1 = os.path.join(save_path, name1)

            img_lab_unet = Variable(img_lab_unet).cuda()
            img_lab_vgg = Variable(pic_vgg).cuda()
            img_lab = img_lab.permute(0, 3, 1, 2)
            img_lab = Variable(img_lab).cuda()

            gray_L = Variable(gray_L).cuda()
            gray_A = Variable(gray_A).cuda()
            gray_B = Variable(gray_B).cuda()

            mean = mean.cuda()
            std = std.cuda()

            parm_dev_mean, parm_mt, parm_six, pre_rgb = net(img_lab_unet, img_lab_vgg, img_lab, gray_L, gray_A, gray_B,
                                                            mean, std)

            pre_rgb_mid = pre_rgb.permute(0, 2, 3, 1).squeeze(0)
            pre_pic_numpy = pre_rgb_mid.detach().cpu().numpy()
            pre_pic_final = np.array(pre_pic_numpy, dtype=np.uint8)
            cv2.imwrite(save_name1, pre_pic_final)


def get_args():
    parser = argparse.ArgumentParser(description='Train the CTNet on images')
    parser.add_argument('--dataroot', type=str, required=True, help='get test images')
    parser.add_argument('--name', type=str, default='test', help='name of test model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--input_size', type=tuple, default=(256, 256), help='the input size of test images')
    parser.add_argument('--output_size', type=tuple, default=(256, 256), help='the output size of images')
    parser.add_argument('--num_worker', type=int, default=4, help='threads for loading data')
    parser.add_argument('--model_path', type=str, default='./checkpoint', help='save model')
    parser.add_argument('--results_dir', type=str, default='./results/', help='results save dir')
    parser.add_argument('--suffix', type=str, default='', help='customized suffix, result_dir = name + suffix')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    Test(args)
