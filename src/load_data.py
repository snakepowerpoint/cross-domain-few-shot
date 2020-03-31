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
import timeit


# for mini-imagenet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Pacs(object):
    def __init__(self):
        self.data_path = '/data/common/cross-domain-few-shot/'
        self.data_dict = self._load_data()
    
    def _load_data(self):
        data_path = os.path.join(self.data_path, 'pacs_split.pickle')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    def get_task(self, domain, categories, n_shot=5, n_query=15, size=(224, 224), aug=True, mode='train', target='all'):
        n_way = len(categories)
        support = np.empty((n_way, n_shot, size[0], size[1], 3))
        query = np.empty((n_way, n_query, size[0], size[1], 3))
        for i, category in enumerate(categories):
            num_img = len(self.data_dict[mode][domain][category])
            selected_imgs = random.sample(range(num_img), k=n_shot+n_query)

            if target == 'all':
                s_imgs = self.data_dict[mode][domain][category][selected_imgs[:n_shot]]
                q_imgs = self.data_dict[mode][domain][category][selected_imgs[n_shot:]]
                support[i] = resize_batch_img(s_imgs, size=size, aug=aug)
                query[i] = resize_batch_img(q_imgs, size=size, aug=aug)

            elif target == 'support':
                s_imgs = self.data_dict[mode][domain][category][selected_imgs[:n_shot]]
                support[i] = resize_batch_img(s_imgs, size=size, aug=aug)

            elif target == 'query':
                q_imgs = self.data_dict[mode][domain][category][selected_imgs[n_shot:]]
                query[i] = resize_batch_img(q_imgs, size=size, aug=aug)

        return support, query


class Omniglot(object):
    def __init__(self):
        self.data_path = '/data/common/cross-domain-few-shot/'
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
    def __init__(self, mode='test'):
        self.data_path = '/data/common/cross-domain-few-shot/'
        self.mode = mode
        self.data_dict = self._load_data()
        
    def _load_data(self):
        data_path = os.path.join(self.data_path, 'cub.pickle')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        if self.mode == 'train':
            data_dict.pop('test', None)
        else:
            data_dict.pop('val', None)
            data_dict.pop('train', None)
        
        return data_dict

    def get_task(self, n_way=5, n_shot=5, n_query=16, size=(224, 224), aug=False):
        mode = self.mode
        selected_categories = random.sample(list(self.data_dict[mode].keys()), k=n_way)

        support = np.empty((n_way, n_shot, size[0], size[1], 3))
        query = np.empty((n_way, n_query, size[0], size[1], 3))
        for i, category in enumerate(selected_categories):
            num_img = len(self.data_dict[mode][category])
            selected_imgs = random.sample(range(num_img), k=n_shot+n_query)
            
            s_imgs = self.data_dict[mode][category][selected_imgs[:n_shot]]
            q_imgs = self.data_dict[mode][category][selected_imgs[n_shot:]]

            support[i] = resize_batch_img(s_imgs, size=size, aug=aug)
            query[i] = resize_batch_img(q_imgs, size=size, aug=aug)

        return support, query


class MiniImageNet(object):
    def __init__(self, resize=False):
        self.data_path = '/data/common/cross-domain-few-shot/mini-imagenet'
        self.resize = resize
        self.data_dict = self._load_data()
        self.label_dict = self._label_mapping() # wei, for pretrain baseline training data
        
    def _load_data(self):
        train_data_path = os.path.join(self.data_path, 'mini-imagenet-cache-train.pkl')
        test_data_path = os.path.join(self.data_path, 'mini-imagenet-cache-test.pkl')
        val_data_path = os.path.join(self.data_path, 'mini-imagenet-cache-val.pkl')
        data_dict = {}
        with open(train_data_path, 'rb') as f:
            data_dict['train'] = pickle.load(f)
        with open(test_data_path, 'rb') as f:
            data_dict['test'] = pickle.load(f)
        with open(val_data_path, 'rb') as f:
            data_dict['val'] = pickle.load(f)

        if self.resize:
            data_dict = self._resize(data_dict)
            data_dict = self._normalized(data_dict)
        return data_dict
    
    def _label_mapping(self):
        # make label mapping list 
        label_dict = dict()
        category_list = list(self.data_dict['train']['class_dict'].keys())
        for i, category in enumerate(category_list):
            label_dict[category] = i

        return label_dict
    
    def _normalized(self, data_dict):
        data_dict['train']['image_data'] = data_dict['train']['image_data'] / 255.0 
        data_dict['val']['image_data'] = data_dict['val']['image_data'] / 255.0
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
    
    def get_task(self, n_way=5, n_shot=5, n_query=16, size=(224, 224), aug=True, mode='train', target='all'):
        support = np.empty((n_way, n_shot, size[0], size[1], 3))
        query = np.empty((n_way, n_query, size[0], size[1], 3))
        selected_categories = random.sample(list(self.data_dict[mode]['class_dict'].keys()), k=n_way)

        for i, category in enumerate(selected_categories):
            selected_imgs = random.sample(
                self.data_dict[mode]['class_dict'][category], k=n_shot+n_query)

            if target == 'all':
                s_imgs = self.data_dict[mode]['image_data'][selected_imgs[:n_shot]]
                q_imgs = self.data_dict[mode]['image_data'][selected_imgs[n_shot:]]
                support[i] = resize_batch_img(s_imgs, size=size, aug=aug)
                query[i] = resize_batch_img(q_imgs, size=size, aug=aug)

            elif target == 'support':
                s_imgs = self.data_dict[mode]['image_data'][selected_imgs[:n_shot]]
                support[i] = resize_batch_img(s_imgs, size=size, aug=aug)
                
            elif target == 'query':
                q_imgs = self.data_dict[mode]['image_data'][selected_imgs[n_shot:]]
                query[i] = resize_batch_img(q_imgs, size=size, aug=aug)

        return support, query
    
    def get_batch(self, batch_size=64, size=(224, 224), aug=True, mode='train'):
        if batch_size % 64 != 0:
            print(">>> batch_size = {} should be divisible by 64.".format(batch_size))
            return None, None
        
        img_batch = np.empty((batch_size, 84, 84, 3), dtype=np.uint8)
        img_label = np.empty((batch_size, 64), dtype=np.uint8)
        img_count = 0
        for category in list(self.data_dict['train']['class_dict'].keys()):     
            # wei, "train" and "val" contrain different classes, and here, we only use images from "train" data to do the validation.
            # Hence, the mode is "val" though, we still get the data from "train".
            if mode == 'train':      
                selected_imgs = random.sample(self.data_dict[mode]['class_dict'][category][:500], k=(batch_size//64))            
            elif mode == 'val':   
                selected_imgs = random.sample(self.data_dict['train']['class_dict'][category][501:], k=(batch_size//64))            
            
            # wei, get the label of the current class from the dict.
            curr_label = np.zeros((1, 64))
            curr_label[0, self.label_dict[category]] = 1

            for j in selected_imgs:
                s_img = self.data_dict['train']['image_data'][j]
                img_batch[img_count] = s_img
                img_label[img_count] = curr_label
                img_count += 1
            
        img_batch = resize_batch_img(img_batch, size=size, aug=aug)

        return img_batch, img_label

def resize_batch_img(imgs, size, aug):       
    resized_img = np.empty((imgs.shape[0], size[0], size[1], 3))
    for i, i_img in enumerate(imgs):
        
        if aug:
            i_img = augmentation(i_img, size=84)
            #resized_img[i] = cv2.resize(i_img, size) # wei, cv2 will change value range
            resized_img[i] = i_img
        else:
            i_img = center_crop(i_img, size=size)
            i_img = ((i_img / 255.0) - mean) / std
            resized_img[i] = i_img
        
    return resized_img


def augmentation(img, size):
    def jitter(img):
        _transform_dict = {'brightness': 0.4, 'contrast': 0.4, 'color': 0.4}
        _color_jitter = ColorJitter(_transform_dict)
        
        img = _color_jitter(img)
        #img = np.array(img) # wei, no need
        return img

    def random_horizontal_flip(img):
        # image shape: [h, w, 3]  ***[begin:end:step]
        if bool(random.getrandbits(1)):
            return img[:, ::-1, :]
        else:
            return img

    def normalize(img):
        return ((img / 255.0) - mean) / std
    
    img = Image.fromarray(img)
    _random_resized_crop = RandomResizedCrop(size=size)
    img = _random_resized_crop(img)
    img = jitter(img)
    img = np.array(img.resize((224, 224))) # wei, use PIL lib to resize here
    img = random_horizontal_flip(img)
    img = normalize(img)
    return img


def center_crop(img, size):
    height, width = size
    new_height, new_width = [int(height*1.15), int(width*1.15)] # wei
    
    img = Image.fromarray(img)
    img = img.resize((new_width, new_height))
    
    left = (new_width - width)/2
    top = (new_height - height)/2
    right = (new_width + width)/2
    bottom = (new_height + height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    img = img.resize((width, height))
    return np.array(img)
