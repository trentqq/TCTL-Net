# -- coding: utf-8 --
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from data.aligned_dataset import TrainDataset
import argparse
import os
from utils.ssim_loss import SSIM
from model.TCTLNet_model import TCTLNet


def train(args):
    iter_count = 0

    input_data = TrainDataset(args.dataroot, output_size=args.output_size)
    input_dataloader = DataLoader(input_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using {} device.'.format(device))

    net = TCTLNet(output_size=args.output_size, init_weights=True)
    net.to(device)

    if args.continue_train:
        net.load_state_dict(torch.load(os.path.join(args.model_path, args.name, 'net_latest.pth')))

    save_path = os.path.join(args.model_path, args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    L1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    ssim_loss = SSIM()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    print('start training:{}'.format(args.name))
    for epoch in range(args.epochs):
        net.train()

        for step, (input_for_unet, input_for_vgg, raw_pic_for_trans, refer_img, true_num, gray_L, gray_A, gray_B, mean,
                   std) in enumerate(input_dataloader):

            input_for_unet_mid = Variable(input_for_unet).cuda()
            input_for_vgg_mid = Variable(input_for_vgg).cuda()

            raw_pic_for_trans_mid = raw_pic_for_trans.permute(0, 3, 1, 2)
            raw_pic_for_trans_final = Variable(raw_pic_for_trans_mid).cuda()

            refer_img = refer_img.permute(0, 3, 1, 2)
            refer_img = refer_img.cuda()

            gray_L = Variable(gray_L).cuda()
            gray_A = Variable(gray_A).cuda()
            gray_B = Variable(gray_B).cuda()

            mean_final = mean.cuda()
            std_final = std.cuda()

            true_num_mid = torch.stack(true_num, dim=1)
            true_num_final = true_num_mid.cuda()

            parm_dev_mean, parm_final_mt, parm_six, Color_transed_img = net(input_for_unet_mid, input_for_vgg_mid,
                                                                            raw_pic_for_trans_final, gray_L, gray_A,
                                                                            gray_B, mean_final, std_final)

            l1_loss = L1_loss(parm_six, true_num_final.float())
            ssim = (1 - ssim_loss(Color_transed_img, refer_img))
            mse = mse_loss(Color_transed_img, refer_img)
            total_loss = 10.0 * l1_loss + ssim * 100.0 + mse * 0.1

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if iter_count % args.print_freq == 0:
                print(
                    'step:{}, train epoch[{}/{}] total_loss:{}  l1_loss:{} mse:{} '.format(step, epoch + 1, args.epochs,
                                                                                           total_loss, l1_loss * 10.0,
                                                                                           mse * 0.1));
                iter_count += 1

        if epoch % args.save_freq == 0:
            print('save Training model epoch' + str(epoch))
            torch.save(net.state_dict(), os.path.join(save_path, 'net_idx' + str(epoch) + '.pth'))
            torch.save(net.state_dict(), os.path.join(save_path, 'net_latest.pth'))

    print('Finished Training')
    torch.save(net.state_dict(), os.path.join(save_path, 'net_latest.pth'))


def get_args():
    parser = argparse.ArgumentParser(description='Train the CTNet on images')
    parser.add_argument('--dataroot', required=True,
                        help='path to images and txts (should have subfolders raw, ref and txt)')
    parser.add_argument('--name', type=str, default='experiment_name', help='name of experiment')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output_size', type=tuple, default=(256, 256), help='output image size')
    parser.add_argument('--dev', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=400, help='train epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='threads for loading data')
    parser.add_argument('--model_path', type=str, default='./checkpoint', help='save model')
    parser.add_argument('--continue_train', action='store_true', help='continue training')
    parser.add_argument('--print_freq', type=int, default=20, help='loss print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='model save frequency')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    train(args)
