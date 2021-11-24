from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os, cv2, shutil
import skimage.io as io
import skimage.transform as trans


def adjustData(img,mask):
    if(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def scale(input_folder, output_folder, scale_percent):
    for file in sorted(os.listdir(input_folder)):
        if '.png' in file:
            tmp = cv2.imread(input_folder + '/' + file, 0)
            width = int(tmp.shape[1] * scale_percent / 100)
            height = int(tmp.shape[0] * scale_percent / 100)
            dim = (width, height)
            tmp = cv2.resize(tmp, dim, interpolation = cv2.INTER_AREA)
            io.imsave(output_folder + '/' + file, tmp)
    print('Scaling is done.')

def pad(input_folder, output_folder,already_padded):
    for file in sorted(os.listdir(input_folder)):
        if '.png' in file:
            tmp = cv2.imread(input_folder + '/' + file, 0)
            width = int(tmp.shape[1])
            height = int(tmp.shape[0])
            wantedWidth = width
            wantedHeight = height
            if width % 32 != 0:
                wantedWidth = width + 32 - width % 32
            if height % 32 != 0:
                wantedHeight = height + 32 - height % 32
            top= int(np.ceil((wantedHeight-height) / 2))
            bottom= int(np.floor((wantedHeight-height) / 2))
            left = int(np.ceil((wantedWidth-width) / 2))
            right = int(np.floor((wantedWidth-width) / 2))
            if not already_padded:
                tmp = cv2.copyMakeBorder(tmp.copy(), top, bottom, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))
            io.imsave(output_folder + '/' + file, tmp)
    if not already_padded:
        print('Padding is done.')


def crop(input_folder, output_folder):
    for file in sorted(os.listdir(input_folder)):
        if '.png' in file:
            tmp = cv2.imread(input_folder + '/' + file, 0)
            tmp = tmp[12:-12, 5:-6]
            io.imsave(output_folder + '/' + file, tmp)
    print('Cropping is done.')


def threshold(folder):
    for img in sorted(os.listdir(folder)):
        tmp = cv2.imread(folder + '/' + img, 0)
        _, tmp = cv2.threshold(tmp,70,255,cv2.THRESH_BINARY)
        io.imsave(folder + '/' + img, tmp)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    save_to_dir = None,target_size = (608,576),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask)
        yield (img,mask)



def testGenerator(test_path,target_size = (608,576),as_gray = True):
    for file in os.listdir(test_path):
        img = io.imread(os.path.join(test_path,file),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,))
        img = np.reshape(img,(1,)+img.shape)
        yield img



def saveResult(save_path,npyfile):
    shutil.rmtree(save_path, ignore_errors=True, onerror=None)
    os.mkdir(save_path)
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)