'''
It is used to preprocess the data, the data comes from kaggle dataset!
The data is not concluded in the project. Therefore, if you just want to run AIRadiologist, you should not run this file!
The dataset is different from the data in preprocessing.py, with contains 704 normal pictures and 700 tuberculosis pictures.
'''
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
root_path = r"Data/Chest Xray Masks and Labels/"
png_path_normal = r"Data/TB2/Normal"
png_path_tub = r"Data/TB2/Tuberculosis"
out_path_normal = r"Data/TB2/Normal"
out_path_tub = r"Data/TB2/Tuberculosis"
mask_path = r"data1_masks"
out_path_mask_normal = r"Data/TB2/Normal"
out_path_mask_tub = r"Data/TB2/Tuberculosis"

def getoverMask(png_path_normal, png_path_tub):
    '''

    :param png_path_normal: path of normal pictures
    :param png_path_tub: path of tuberculosis pictures
    :return: pictures of chest x-ray covered by masks
    '''
    for root, dirs, files in os.walk(png_path_normal):
        for file in files:
            try:
                img = mpimg.imread(os.path.join(root, file))
                img.flags.writeable = True
                mask = mpimg.imread(os.path.join(mask_path, file))
                # change to image from array
                Image.fromarray(np.uint8(img))
                # pixels become to 255 when the pixels are not in the mask area
                img[:, :][mask[:, :] == 0] = 255
                # pixels change from grayscale to RGB scale when the pixels are in the mask
                img[:, :][mask[:, :] == 1] *= 255

                cv2.imwrite(os.path.join(out_path_mask_normal,file),img)
                print(file)
                # get exception when the file exists!
            except FileExistsError:
                print("FILE EXISTS")
            # get exception when the file cannot be found!
            except FileNotFoundError:
                print("NO SUCH FILE")

    for root, dirs, files in os.walk(png_path_tub):
        for file in files:
            try:
                img = mpimg.imread(os.path.join(root, file))
                img.flags.writeable = True
                mask = mpimg.imread(os.path.join(mask_path, file))
                Image.fromarray(np.uint8(img))
                img[:, :][mask[:, :] == 0] = 255
                img[:, :][mask[:, :] == 1] *= 255

                cv2.imwrite(os.path.join(out_path_mask_tub,file),img)
                print(file)
            except FileExistsError:
                print("FILE EXISTS")
            except FileNotFoundError:
                print("NO SUCH FILE")
getoverMask(png_path_normal,png_path_tub)