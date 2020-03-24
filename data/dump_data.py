# io
import os
import cv2
from PIL import Image
import pickle
import argparse
import scipy.misc

# computation
import numpy as np
import random


# from mini-imagenet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)


def augmentation(img):
    img_rot90 = np.rot90(img, k=1)
    img_rot270 = np.rot90(img, k=3)
    img_flip_h = horizontal_flip(img)
    img_flip_v = vertical_flip(img)
    img_aug = np.stack([img, img_rot90, img_rot270, img_flip_h, img_flip_v])
    return img_aug


def horizontal_flip(img):
    # image shape: [32, 32, 3]  ***[begin:end:step]
    img_flip = img[:, ::-1, :]
    return img_flip


def vertical_flip(img):
    # image shape: [32, 32, 3]
    img_flip = img[::-1, :, :]
    return img_flip


def resize(img, size):
    height, width = size
    img_resized = cv2.resize(img, (width, height))
    return img_resized


class Pacs(object):
    def __init__(self):
        self.domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        self.categories = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.data_path = '/data/common/cross-domain-few-shot/pacs'

    def load_data(self, is_aug=True, is_normalize=True):
        data_dict = {}        
        for domain in self.domains:
            data_dict[domain] = {}
            domain_path = os.path.join(self.data_path, domain)
            for category in self.categories:
                data_dict[domain][category] = []
                category_path = os.path.join(domain_path, category)
                img_files = os.listdir(category_path)
                for img_file in img_files:
                    path = os.path.join(category_path, img_file)
                    img = scipy.misc.imread(path, mode='RGB').astype(np.uint8)
                    img = Image.fromarray(img)
                    img = np.array(img.resize((224, 224)))
                    img = np.expand_dims(img, axis=0)
                    data_dict[domain][category].append(img)
                data_dict[domain][category] = np.concatenate(data_dict[domain][category])      
        return data_dict
    

class Cub200(object):
    def __init__(self):
        self.data_path = '/data/common/cross-domain-few-shot/cub200/CUB_200_2011/images'   

    def load_data(self):
        data_dict = {}
        data_dict['train'] = {}
        data_dict['val'] = {}
        data_dict['test'] = {}
        
        folder_list = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)]
        random.shuffle(folder_list)
        
        for dataset in list(data_dict.keys()):
            for i, folder in enumerate(folder_list):
                if 'train' in dataset:
                    if (i % 2 == 0):
                        file_list = [os.path.join(folder, f) for f in os.listdir(folder)]
                        random.shuffle(file_list)

                        data_dict[dataset][i] = []
                        for img_path in file_list:
                            img = scipy.misc.imread(img_path, mode='RGB').astype(np.uint8)
                            img = Image.fromarray(img)
                            img = np.array(img.resize((224, 224)))
                            data_dict[dataset][i].append(img)
                        data_dict[dataset][i] = np.stack(data_dict[dataset][i])

                if 'val' in dataset:
                    if (i % 4 == 1):
                        file_list = [os.path.join(folder, f) for f in os.listdir(folder)]
                        random.shuffle(file_list)

                        data_dict[dataset][i] = []
                        for img_path in file_list:
                            img = scipy.misc.imread(img_path, mode='RGB').astype(np.uint8)
                            img = Image.fromarray(img)
                            img = np.array(img.resize((224, 224)))
                            data_dict[dataset][i].append(img)
                        data_dict[dataset][i] = np.stack(data_dict[dataset][i])

                if 'test' in dataset:
                    if (i % 4 == 3):
                        file_list = [os.path.join(folder, f) for f in os.listdir(folder)]
                        random.shuffle(file_list)

                        data_dict[dataset][i] = []
                        for img_path in file_list:
                            img = scipy.misc.imread(img_path, mode='RGB').astype(np.uint8)
                            img = Image.fromarray(img)
                            img = np.array(img.resize((224, 224)))
                            data_dict[dataset][i].append(img)
                        data_dict[dataset][i] = np.stack(data_dict[dataset][i])

        return data_dict

        

args = parser.parse_args()
if args.data == 'pacs':
    pacs = Pacs()
    data = pacs.load_data()

    f = open('pacs.pickle', 'wb')
    pickle.dump(data, f)
    f.close()

if args.data == 'cub':
    cub = Cub200()
    data = cub.load_data()

    f = open('cub.pickle', 'wb')
    pickle.dump(data, f)
    f.close()

