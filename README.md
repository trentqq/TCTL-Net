# TCTL-Net: Template-free Color Transfer Learning for Self-Attention Driven Underwater Image Enhancement
This repository is the official PyTorch implementation of TCTL-Net.
## Dataset preparation 
- For training, you need to prepare the **raw**, **ref**, and **txt** folders in the train data root directory to store the raw degraded image, the reference enhanced image, and the reference true value of color-transfer, respectively. 
- For testing, you need to prepare the **raw** folder in the test data root directory to store the test images.

## Train
``` 
python train.py --dataroot /path_to_data --name train_name
```
## Test
```
python test.py --dataroot /path_to_data --name test_name
```
You can download the trained model from [Baidu Netdisk](https://pan.baidu.com/s/1zLHssfNI2sgXMSb6ybyQfw)(Extraction code:rw85) or [Google Drive](https://drive.google.com/file/d/1OAESvi-uIRfELIbX_dLlas43Lx-USxvf/view?usp=sharing).