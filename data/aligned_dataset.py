# -- coding: utf-8 --import osimport torchfrom torch.utils.data.dataset import Datasetfrom torchvision import transformsfrom tqdm import tqdmimport numpy as npimport cv2from torch.utils.data.dataloader import DataLoaderfrom utils.util import get_gray, trans_np'''train tpye_I:class TrainDataset：input to unet and vgg in normalize range:L(0, 100), A(-128, 127),B(-128, 127), L attention map:I=(0, 100)->I'=I/100->I"=1-I'A attention map:I=(-128, 127)->I=I+128->I'=I/255->I"=1-I'B attention map:I=(-128, 127)->I=I+128->I'=I/255->I"=1-I''''class TrainDataset(Dataset):    def __init__(self, dataroot, output_size=(256, 256)):        # 1.get raw image path, ref txt path and ref iamges path        self.raw_path = os.path.join(dataroot, 'raw_pic')        self.tr_txt_path = os.path.join(dataroot, 'ref_txt')        self.tr_images_path = os.path.join(dataroot, 'ref_pic')        self.output_size = output_size        # 2.set box for raw images, ref images and its name ,ect on        self.raw_images = []        self.tr_pics = []        self.tr_num = {}        self.pic_name = []        self.transform = transforms.Compose([transforms.ToTensor()])        pics = os.listdir(self.raw_path)        # 3.add pic        for pic in pics:            self.raw_images.append(os.path.join(self.raw_path, pic))            self.tr_pics.append(os.path.join(self.tr_images_path, pic))            self.pic_name.append(pic)            name, type = os.path.splitext(pic)            txt = name + '.txt'            true_num_path = os.path.join(self.tr_txt_path, txt)            with open(true_num_path) as f:                for line in f:                    words = line.split(',')                    # to num list                    word = map(float, words)                    num = list(word)                    self.tr_num[pic] = num    def __getitem__(self, index):        # 4.get raw images, 1)to 256*256, 2)to lab, transform, 3)to tensor, 4)raw_pic_for_trans is input of get_grey,get pic :raw pic to unet, raw pic to vgg        raw_img = cv2.imread(self.raw_images[index])        # 5.deal with images as input of regression net        raw_img_mid_vgg = cv2.resize(raw_img, (256, 256))        raw_img_f_vgg = raw_img_mid_vgg * (1. / 255)        img_lab_vgg = cv2.cvtColor(np.float32(raw_img_f_vgg), cv2.COLOR_BGR2LAB)        # 6.deal with images as input of l a b branch net        raw_img_mid_unet = cv2.resize(raw_img, self.output_size)        raw_img_f_unet = raw_img_mid_unet * (1. / 255)        img_lab_unet = cv2.cvtColor(np.float32(raw_img_f_unet), cv2.COLOR_BGR2LAB)        # 7.get attention map        gray_L, gray_A, gray_B = get_gray(img_lab_unet)        input_for_vgg = self.transform(img_lab_vgg)        input_for_unet = self.transform(img_lab_unet)        raw_pic_for_trans = torch.from_numpy(img_lab_unet).float()        # 8.true_pic_for_trans is refer image in bgr        tr_img = cv2.imread(self.tr_pics[index])        tr_img_givensize = cv2.resize(tr_img, self.output_size)        true_pic_for_trans = torch.from_numpy(tr_img_givensize).float()        mean, std = trans_np(img_lab_unet)        mean_final = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(mean), -1), -1)        std_final = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(std), -1), -1)        true_num = self.tr_num[self.pic_name[index]]        # get orginal size output images        return input_for_unet, input_for_vgg, raw_pic_for_trans, true_pic_for_trans, true_num, gray_L, gray_A, gray_B, mean_final, std_final    def __len__(self):        return len(self.raw_images)class TestDataset():    def __init__(self, dataroot, input_size, output_size):        self.test_path = os.path.join(dataroot, 'raw_pic')        # self.test_ref_path = opt.val_tr_pic_path        self.test_images = []        self.test_ref_images = []        self.pic_name = []        self.transform = transforms.Compose([transforms.ToTensor()])        self.input_size = input_size        self.output_size = output_size        pics = os.listdir(self.test_path)        for pic in pics:            self.test_images.append(os.path.join(self.test_path, pic))            # self.test_ref_images.append(os.path.join(self.test_ref_path, pic))            self.pic_name.append(pic)    def __getitem__(self, index):        # 0.get raw images        raw_img = cv2.imread(self.test_images[index])        raw_img_1 = cv2.resize(raw_img, self.output_size)        # 1.get input-vgg        raw_img_input = cv2.resize(raw_img_1, (256, 256))        raw_img_input1 = raw_img_input * (1. / 255)        img_lab_vgg = cv2.cvtColor(np.float32(raw_img_input1), cv2.COLOR_BGR2LAB)        # 2.get input-unet        raw_img_givensize1_unet = raw_img_input * (1. / 255)        img_lab_unet = cv2.cvtColor(np.float32(raw_img_givensize1_unet), cv2.COLOR_BGR2LAB)        # 3.get raw images for color transfer        raw_img_for_trans1 = raw_img_1 * (1. / 255)        img_lab_raw = cv2.cvtColor(np.float32(raw_img_for_trans1), cv2.COLOR_BGR2LAB)        gray_L, gray_A, gray_B = get_gray(img_lab_unet)        mean, std = trans_np(img_lab_raw)        mean1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(mean), -1), -1)        std1 = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(std), -1), -1)        pic_for_unet = self.transform(img_lab_unet)        pic_for_vgg = self.transform(img_lab_vgg)        pic_name = str(self.pic_name[index])        raw_pic_for_trans = torch.from_numpy(img_lab_raw).float()        # true_pic_for_trans1  = true_pic_for_trans.permute(0,3,1,2)        return pic_for_unet, pic_for_vgg, pic_name, raw_pic_for_trans, gray_L, gray_A, gray_B, mean1, std1    def __len__(self):        return len(self.test_images)if __name__ == '__main__':    train_dataset_path = '/data/fanht/dataset/NYU_all/NYU/train/raw_pic'    ref_pic_path = '/data/fanht/dataset/NYU_all/NYU/train/ref_pic'    ref_txt_path = '/data/fanht/dataset/NYU_all/NYU/train/ref_txt'    input_data = Cosole(train_dataset_path, ref_pic_path, ref_txt_path, output_size=(256, 256))    input_dataloader = DataLoader(input_data, batch_size=8, shuffle=True, num_workers=4)    train_bar = tqdm(input_dataloader)    for step, (input_for_unet, raw_img_givensize) in enumerate(train_bar):        print(input_for_unet, raw_img_givensize)