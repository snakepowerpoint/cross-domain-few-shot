# io
import os
import pickle
import cv2
from PIL import Image

# computation
import numpy as np
import random
import scipy.misc
from .color_jitter import ColorJitter, RandomResizedCrop


class Pacs(object):
    def __init__(self):
        self.data_path = 'data/'
        self.data_dict = self._load_data()
    
    def _load_data(self):
        data_path = os.path.join(self.data_path, 'pacs.pickle')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    def get_task(self, domain, categories, n_shot=5, n_query=15):
        support = []
        query = []
        for category in categories:
            num_img = len(self.data_dict[domain][category])
            selected_imgs = random.sample(range(num_img), k=n_shot+n_query)
            support.append(self.data_dict[domain][category][selected_imgs[:n_shot]])
            query.append(self.data_dict[domain][category][selected_imgs[n_shot:]])

        support = np.stack(support)
        query = np.stack(query)  
        return support, query


class Omniglot(object):
    def __init__(self):
        self.data_path = 'data/'
        self.data_dict = self._load_data()

    def _load_data(self):
        data_path = os.path.join(self.data_path, 'omniglot.pickle')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        data_dict = self._normalize(data_dict)
        data_dict = self._concat(data_dict)
        return data_dict

    def _normalize(self, data_dict):
        for split in list(data_dict.keys()):
            for alphabet in list(data_dict[split].keys()):
                for char in list(data_dict[split][alphabet].keys()):
                    data_dict[split][alphabet][char] = data_dict[split][alphabet][char] / 255.0
        return data_dict

    def _concat(self, data_dict):
        new_data_dict = {}
        for split in list(data_dict.keys()):
            new_data_dict[split] = {}
            for alphabet in list(data_dict[split].keys()):
                for char in list(data_dict[split][alphabet].keys()):
                    new_data_dict[split][alphabet+'_'+char] = data_dict[split][alphabet][char]
        return new_data_dict

    def get_task(self, n_way, n_shot, n_query, mode='train'):
        if mode == 'train':
            split = 'background'
        if mode == 'test':
            split = 'evaluation'

        chars = random.sample(list(self.data_dict[split].keys()), n_way)

        support = []
        query = []
        for char in chars:
            num_data = self.data_dict[split][char].shape[0]
            idx = np.random.choice(np.arange(num_data), size=n_shot+n_query, replace=False)
            support.append(self.data_dict[split][char][idx[:n_shot]])
            query.append(self.data_dict[split][char][idx[n_shot:]])
        
        support = np.stack(support)
        query = np.stack(query)
        return support, query


class Cub(object):
    def __init__(self, size, mode='test'):
        self.data_path = 'data/'
        self.mode = mode
        self.data_dict = self._load_data(size=size)
        
    def _load_data(self, size):
        data_path = os.path.join(self.data_path, 'cub_crop.pickle')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        if self.mode == 'train':
            data_dict.pop('test', None)
        else:
            data_dict.pop('train', None)
        
        data_dict = self._resize(data_dict, size=size)
        return data_dict

    def _resize(self, data_dict, size):
        h, w = size
        for dataset, class_dict in data_dict.items():
            for class_name, image_list in class_dict.items():
                data_dict[dataset][class_name] = []
                for image in image_list:
                    resized_img = cv2.resize(image, (h, w))
                    data_dict[dataset][class_name].append(resized_img)
                data_dict[dataset][class_name] = np.stack(data_dict[dataset][class_name])
        return data_dict

    def get_task(self, n_way=5, n_shot=5, n_query=15):
        mode = self.mode
        support = []
        query = []
        selected_categories = random.sample(list(self.data_dict[mode].keys()), k=n_way)
        for category in selected_categories:
            num_img = self.data_dict[mode][category].shape[0]
            selected_imgs = random.sample(range(num_img), k=n_shot+n_query)
            support.append(self.data_dict[mode][category][selected_imgs[:n_shot]])
            query.append(self.data_dict[mode][category][selected_imgs[n_shot:]])

        support = np.stack(support)
        query = np.stack(query)
        return support, query


class MiniImageNet(object):
    def __init__(self, resize=False):
        self.data_path = 'data/mini-imagenet'
        self.resize = resize
        self.data_dict = self._load_data()
        
    def _load_data(self):
        train_data_path = os.path.join(self.data_path, 'mini-imagenet-cache-train.pkl')
        test_data_path = os.path.join(self.data_path, 'mini-imagenet-cache-test.pkl')
        data_dict = {}
        with open(train_data_path, 'rb') as f:
            data_dict['train'] = pickle.load(f)
        with open(test_data_path, 'rb') as f:
            data_dict['test'] = pickle.load(f)
        
        if self.resize:
            data_dict = self._resize(data_dict)
            data_dict = self._normalized(data_dict)
        return data_dict
    
    def _normalized(self, data_dict):
        data_dict['train']['image_data'] = data_dict['train']['image_data'] / 255.0 
        data_dict['test']['image_data'] = data_dict['test']['image_data'] / 255.0
        return data_dict
    
    def _resize(self, data_dict, size=(224, 224)):
        h, w = size
        def resize_img(img):
            return cv2.resize(img, (h, w))

        for dataset, _ in data_dict.items():
            image_data = []
            for img in data_dict[dataset]['image_data']:
                image_data.append(resize_img(img))
            data_dict[dataset]['image_data'] = np.stack(image_data)
        return data_dict
    
    def get_task(self, n_way=5, n_shot=5, n_query=16, size=(224, 224), aug=True, mode='train'):
        support = []
        query = []
        selected_categories = random.sample(list(self.data_dict[mode]['class_dict'].keys()), k=n_way)
        for category in selected_categories:
            selected_imgs = random.sample(
                self.data_dict[mode]['class_dict'][category], k=n_shot+n_query)
            
            s_imgs = self.data_dict[mode]['image_data'][selected_imgs[:n_shot]]
            q_imgs = self.data_dict[mode]['image_data'][selected_imgs[n_shot:]]
            support.append(self.resize_batch_img(s_imgs, size=size, aug=aug))
            query.append(self.resize_batch_img(q_imgs, size=size, aug=aug))

        support = np.stack(support)
        query = np.stack(query)
        return support, query
    
    def resize_batch_img(self, imgs, size, aug):
        resized_img = []
        for i_img in imgs:
            img = cv2.resize(i_img, size)
            if aug:
                resized_img.append(augmentation(img, size=size))
            else:
                resized_img.append(img / 255.0)
        return np.stack(resized_img)


def augmentation(img, size):
    def jitter(img):
        _transform_dict = {'brightness': 0.4, 'contrast': 0.4, 'color': 0.4}
        _color_jitter = ColorJitter(_transform_dict)
        
        img = _color_jitter(img)
        img = np.array(img)
        return img

    def random_horizontal_flip(img):
        # image shape: [h, w, 3]  ***[begin:end:step]
        if bool(random.getrandbits(1)):
            return img[:, ::-1, :]
        else:
            return img

    def normalize(img):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return ((img / 255.0) - mean) / std
    
    img = Image.fromarray(img)
    _random_resized_crop = RandomResizedCrop(size=size)
    img = _random_resized_crop(img)
    img = jitter(img)
    img = random_horizontal_flip(img)
    img = normalize(img)
    return img
