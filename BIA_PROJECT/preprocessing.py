'''
It is used to preprocess the data, the data comes from kaggle dataset!
The data is not concluded in the project. Therefore, if you just want to run AIRadiologist, you should not run this file!
'''

import os
import shutil
import cv2
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from distributed.protocol import torch
from mpmath.identification import transforms
from prediction import segmentation
from segmentation_gpu import UNet

root_path = r"Data/Chest Xray Masks and Labels/"
png_path = r"Data/Chest Xray Masks and Labels/CXR_png"
out_path = r"Data/Chest Xray Masks and Labels/TB_Chest_Radiography_Database"
out_path_normal = r"Data/Chest Xray Masks and Labels/TB_Chest_Radiography_Database/Normal"
out_path_tub = r"Data/Chest Xray Masks and Labels/TB_Chest_Radiography_Database/Tuberculosis"
mask_path = r"Data/Chest Xray Masks and Labels/masks"
out_path_mask = r"Data/Chest Xray Masks and Labels/CXR_png_mask"

png_path_normal = r"Data/TB2/Normal"
png_path_tub = r"Data/TB2/Tuberculosis"

# open the working path
def getoverMask(png_path):
    '''
    :param png_path:
    :return: pictures covered by masks
    Use the mask to cover the original picture.
    we can get pictures only available in the mask area.
    '''
    for root, dirs, files in os.walk(png_path):
        for file in files:
            try:
                img = mpimg.imread(os.path.join(root, file))
                img.flags.writeable = True
                mask_file = file.split(".")[0]+"_mask."+file.split(".")[1]
                mask = mpimg.imread(os.path.join(mask_path, mask_file))
                # change to image from array
                Image.fromarray(np.uint8(img))
                # pixels become to 255 when the pixels are not in the mask area
                img[:, :][mask[:, :] == 0] = 255
                # pixels change from grayscale to RGB scale when the pixels are in the mask
                img[:, :][mask[:, :] == 1] *= 255

                cv2.imwrite(os.path.join(out_path_mask,file),img)
                print(file)
            # get exception when the file exists!
            except FileExistsError:
                print("FILE EXISTS")
            # get exception when the file cannot be found!
            except FileNotFoundError:
                print("NO SUCH FILE")
getoverMask(png_path)

def getCategory(out_path_mask):
    '''
    :param out_path_mask:
    :return: folders with category
    It is categorized by file name in mask. 0 represents Normal and 1 represents Tuberculosis
    '''
    for root, dirs, files in os.walk(out_path_mask):
        for file in files:
            try:
                if file.split('_')[2].split(".")[0] == "0":
                    # copy the pictures in normal category
                    shutil.copy(os.path.join(root, file), os.path.join(out_path_normal ,file))
                elif file.split('_')[2].split(".",)[0] == "1":
                    # copy the pictures in Tuberculosis category
                    shutil.copy(os.path.join(root, file), os.path.join(out_path_tub , file))
                print(file)
            except FileNotFoundError:
                print("NO SUCH FILE")
getCategory(out_path_mask)

png_path_normal = r"Data/TB_Chest_Radiography_Database/Normal"
png_path_tub = r"/Data/TB_Chest_Radiography_Database/Tuberculosis"

def predictionMasks(png_path_normal, png_path_tub):
    '''
    :param png_path_normal: path of normal chest x-ray
    :param png_path_tub: path of tuberculosis chest x-ray
    :return: Segmentation the masks with Unet
    '''
    RESCALE_SIZE = (512, 512)
    FIRST_OUT_CHANNELS = 16
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create unet model to segment the masks
    net = UNet(n_channels=1, n_classes=2, first_out_channels=FIRST_OUT_CHANNELS, bilinear=False).to(DEVICE)
    net.load_state_dict(torch.load(r'Model/unet.pt'))
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.ToTensor(),
                                            transforms.Resize(RESCALE_SIZE)])

    # segment normal pictures
    for root, dirs, files in os.walk(png_path_normal):
        for file in files:
            image_path = os.path.join(root,file)
            image = Image.open(image_path)
            image = transform(image) / 255
            image = image.to(DEVICE)
            image = torch.unsqueeze(image, dim=0)
            # segment image and return the masks
            out = segmentation(image,net,DEVICE)
            # save the masks
            out_path = os.path.join(r'Data1_masks',file)
            out.save(out_path)
            print("Saved" + file)

    # segment tuberculosis pictures
    for root, dirs, files in os.walk(png_path_tub):
        for file in files:
            image_path = os.path.join(root,file)
            image = Image.open(image_path)
            image = transform(image) / 255
            image = image.to(DEVICE)
            image = torch.unsqueeze(image, dim=0)
            out = segmentation(image,net,DEVICE)
            out_path = os.path.join(r'Data1_masks',file)
            out.save(out_path)
            print("Saved" + file)
predictionMasks(png_path_normal,png_path_tub)



