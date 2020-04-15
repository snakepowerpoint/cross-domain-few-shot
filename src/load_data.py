# io
import os
import pickle
import cv2
from PIL import Image
import json

# computation
import numpy as np
import random
import scipy.misc
from .color_jitter import ColorJitter, RandomResizedCrop
import timeit

# identify server mac
import netifaces

from tqdm import tqdm

# for mini-imagenet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Cars(object):
    def __init__(self):
        # self.data_path_base = define_dir_by_mac()
        # self.data_path = self.data_path_base + 'cross-domain-few-shot/'
        self.data_path = '/data/rahul/workspace/cross-domain-few-shot/CrossDomainFewShot/filelists/cars'
        self.meta = self._load_meta()
        
    def _load_meta(self):
        modes = ['base', 'val', 'novel']
        meta = {}
        for mode in modes:
            json_path = os.path.join(self.data_path, mode + '.json')
            with open(json_path, 'r') as f:
                json_file = json.load(f)
                labels = json_file['image_labels']
                meta[mode] = {}
                for i in range(len(labels)):
                    image_name = json_file['image_names'][i]
                    label = labels[i]
                    label_name = json_file['label_names'][label]
                    if label_name not in list(meta[mode].keys()):
                        meta[mode][label_name] = []
                    meta[mode][label_name].append(image_name)
        return meta

    def get_task_from_raw(self, n_way, n_shot, n_query, size=(224, 224), aug=True, mode='train'):
        if mode == 'train':
            split = 'base'
        elif mode == 'val':
            split = 'val'
        elif mode == 'test':
            split = 'novel'
        else:
            raise ValueError('Unknown mode! Please specify mode as either one of train/val/test')
        
        selected_categories = random.sample(list(self.meta[split].keys()), k=n_way)

        support = np.empty((n_way, n_shot, size[0], size[1], 3))
        query = np.empty((n_way, n_query, size[0], size[1], 3))

        for i, category in enumerate(selected_categories):
            selected_imgs_path = random.sample(self.meta[split][category], k=n_shot+n_query)

            for j, curr_img_path in enumerate(selected_imgs_path[:n_shot]):
                curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                support[i][j] = resize_img(curr_img, size=size, aug=aug)

            for j, curr_img_path in enumerate(selected_imgs_path[n_shot:]):
                curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                query[i][j] = resize_img(curr_img, size=size, aug=aug)

        return support, query
        

class Places(object):
    def __init__(self):
        # self.data_path_base = define_dir_by_mac()
        # self.data_path = self.data_path_base + 'cross-domain-few-shot/'
        self.data_path = '/data/rahul/workspace/cross-domain-few-shot/CrossDomainFewShot/filelists/places'
        self.meta = self._load_meta()
    
    def _load_meta(self):
        modes = ['base', 'val', 'novel']
        meta = {}
        for mode in modes:
            json_path = os.path.join(self.data_path, mode + '.json')
            with open(json_path, 'r') as f:
                json_file = json.load(f)
                labels = json_file['image_labels']
                meta[mode] = {}
                for i in range(len(labels)):
                    image_name = json_file['image_names'][i]
                    label = labels[i]
                    label_name = json_file['label_names'][label]
                    if label_name not in list(meta[mode].keys()):
                        meta[mode][label_name] = []
                    meta[mode][label_name].append(image_name)
        return meta

    def get_task_from_raw(self, n_way, n_shot, n_query, size=(224, 224), aug=True, mode='train'):
        if mode == 'train':
            split = 'base'
        elif mode == 'val':
            split = 'val'
        elif mode == 'test':
            split = 'novel'
        else:
            raise ValueError('Unknown mode! Please specify mode as either one of train/val/test')
        
        selected_categories = random.sample(list(self.meta[split].keys()), k=n_way)

        support = np.empty((n_way, n_shot, size[0], size[1], 3))
        query = np.empty((n_way, n_query, size[0], size[1], 3))

        for i, category in enumerate(selected_categories):
            selected_imgs_path = random.sample(self.meta[split][category], k=n_shot+n_query)

            for j, curr_img_path in enumerate(selected_imgs_path[:n_shot]):
                curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                support[i][j] = resize_img(curr_img, size=size, aug=aug)

            for j, curr_img_path in enumerate(selected_imgs_path[n_shot:]):
                curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                query[i][j] = resize_img(curr_img, size=size, aug=aug)

        return support, query


class Plantae(object):
    def __init__(self):
        # self.data_path_base = define_dir_by_mac()
        # self.data_path = self.data_path_base + 'cross-domain-few-shot/'
        self.data_path = '/data/rahul/workspace/cross-domain-few-shot/CrossDomainFewShot/filelists/plantae'
        self.meta = self._load_meta()

    def _load_meta(self):
        modes = ['base', 'val', 'novel']
        meta = {}
        for mode in modes:
            json_path = os.path.join(self.data_path, mode + '.json')
            with open(json_path, 'r') as f:
                json_file = json.load(f)
                labels = json_file['image_labels']
                meta[mode] = {}
                for i in range(len(labels)):
                    image_name = json_file['image_names'][i]
                    label = labels[i]
                    label_name = json_file['label_names'][label]
                    if label_name not in list(meta[mode].keys()):
                        meta[mode][label_name] = []
                    meta[mode][label_name].append(image_name)
        return meta

    def get_task_from_raw(self, n_way, n_shot, n_query, size=(224, 224), aug=True, mode='train'):
        if mode == 'train':
            split = 'base'
        elif mode == 'val':
            split = 'val'
        elif mode == 'test':
            split = 'novel'
        else:
            raise ValueError('Unknown mode! Please specify mode as either one of train/val/test')
        
        selected_categories = random.sample(list(self.meta[split].keys()), k=n_way)

        support = np.empty((n_way, n_shot, size[0], size[1], 3))
        query = np.empty((n_way, n_query, size[0], size[1], 3))

        for i, category in enumerate(selected_categories):
            selected_imgs_path = random.sample(self.meta[split][category], k=n_shot+n_query)

            for j, curr_img_path in enumerate(selected_imgs_path[:n_shot]):
                curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                support[i][j] = resize_img(curr_img, size=size, aug=aug)

            for j, curr_img_path in enumerate(selected_imgs_path[n_shot:]):
                curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                query[i][j] = resize_img(curr_img, size=size, aug=aug)

        return support, query


class Cub(object):
    def __init__(self):
        # self.data_path_base = define_dir_by_mac()
        # self.data_path = self.data_path_base + 'cross-domain-few-shot/'
        self.data_path = '/data/rahul/workspace/cross-domain-few-shot/CrossDomainFewShot/filelists/cub'
        self.meta = self._load_meta()

    def _load_meta(self):
        modes = ['base', 'val', 'novel']
        meta = {}
        for mode in modes:
            json_path = os.path.join(self.data_path, mode + '.json')
            with open(json_path, 'r') as f:
                json_file = json.load(f)
                labels = json_file['image_labels']
                meta[mode] = {}
                for i in range(len(labels)):
                    image_name = json_file['image_names'][i]
                    label = labels[i]
                    label_name = json_file['label_names'][label]
                    if label_name not in list(meta[mode].keys()):
                        meta[mode][label_name] = []
                    meta[mode][label_name].append(image_name)
        return meta

    def get_task_from_raw(self, n_way, n_shot, n_query, size=(224, 224), aug=True, mode='train'):
        if mode == 'train':
            split = 'base'
        elif mode == 'val':
            split = 'val'
        elif mode == 'test':
            split = 'novel'
        else:
            raise ValueError('Unknown mode! Please specify mode as either one of train/val/test')
        
        selected_categories = random.sample(list(self.meta[split].keys()), k=n_way)

        support = np.empty((n_way, n_shot, size[0], size[1], 3))
        query = np.empty((n_way, n_query, size[0], size[1], 3))

        for i, category in enumerate(selected_categories):
            selected_imgs_path = random.sample(self.meta[split][category], k=n_shot+n_query)

            for j, curr_img_path in enumerate(selected_imgs_path[:n_shot]):
                curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                support[i][j] = resize_img(curr_img, size=size, aug=aug)

            for j, curr_img_path in enumerate(selected_imgs_path[n_shot:]):
                curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                query[i][j] = resize_img(curr_img, size=size, aug=aug)

        return support, query


class Pacs(object):
    def __init__(self):
        self.data_path_base = define_dir_by_mac()
        self.data_path = self.data_path_base + 'cross-domain-few-shot/'
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
        self.data_path_base = define_dir_by_mac()
        self.data_path = self.data_path_base + 'cross-domain-few-shot/'
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


class MiniImageNetFull(object):
    def __init__(self):
        self.data_path_base = define_dir_by_mac()
        self.data_path = self.data_path_base + 'cross-domain-few-shot/mini_imagenet_full_size'
        
        self.train_label_mapping, self.train_img_path_mapping = self._get_label_mapping(mode='train')
        self.val_label_mapping, self.val_img_path_mapping = self._get_label_mapping(mode='val')
        self.test_label_mapping, self.test_img_path_mapping = self._get_label_mapping(mode='test')

        self.train_image = self._load_train_image()

    def _load_train_image(self):
        all_categories = sorted(list(self.train_label_mapping.keys()))
        train_image = {}
        for category in all_categories:
            train_image[category] = {}

        print("=== Load Full Size Mini-ImageNet...")
        all_idx_bar = tqdm(range(600*len(all_categories)))
        for idx in all_idx_bar:     
            category = all_categories[idx // 600]
            img_idx = idx % 600

            selected_img_path = self.train_img_path_mapping[category][img_idx]

            s_img = scipy.misc.imread(selected_img_path, mode='RGB').astype(np.uint8)
            train_image[category][img_idx] = s_img
        print(">>> Done.")
        
        return train_image

    def _get_label_mapping(self, mode):
        class_path = os.path.join(self.data_path, mode)
        labels = os.listdir(class_path)
        label_mapping = {}
        img_path_mapping = {}
        for i in range(len(labels)):
            label = labels[i]
            file_names = os.listdir(os.path.join(class_path, label))
            file_paths = [os.path.join(class_path, label, x) for x in file_names]
            label_mapping[label] = i
            img_path_mapping[label] = file_paths
        return label_mapping, img_path_mapping

    def get_task_from_raw(self, n_way=5, n_shot=5, n_query=16, size=(224, 224), aug=True, mode='train'):
        support = np.empty((n_way, n_shot, size[0], size[1], 3))
        query = np.empty((n_way, n_query, size[0], size[1], 3))

        if mode == 'train':
            selected_categories = random.sample(list(self.train_label_mapping.keys()), k=n_way)
            img_path_mapping = self.train_img_path_mapping
        elif mode == 'val':
            selected_categories = random.sample(list(self.val_label_mapping.keys()), k=n_way)
            img_path_mapping = self.val_img_path_mapping
        elif mode == 'test':
            selected_categories = random.sample(list(self.test_label_mapping.keys()), k=n_way)
            img_path_mapping = self.test_img_path_mapping
        else:
            raise ValueError('Unknown mode! Please specify mode as either one of train/val/test.')

        if mode == 'train':
            for i, category in enumerate(selected_categories):
                selected_idx = random.sample(range(len(img_path_mapping[category])), k=n_shot+n_query)

                for j, idx in enumerate(selected_idx[:n_shot]):
                    curr_img = self.train_image[category][idx]
                    support[i][j] = resize_img(curr_img, size=size, aug=aug)
                
                for j, idx in enumerate(selected_idx[n_shot:]):
                    curr_img = self.train_image[category][idx]
                    query[i][j] = resize_img(curr_img, size=size, aug=aug)
        else:
            for i, category in enumerate(selected_categories):
                selected_imgs_path = random.sample(img_path_mapping[category], k=n_shot+n_query)

                for j, curr_img_path in enumerate(selected_imgs_path[:n_shot]):
                    curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                    support[i][j] = resize_img(curr_img, size=size, aug=aug)
                
                for j, curr_img_path in enumerate(selected_imgs_path[n_shot:]):
                    curr_img = scipy.misc.imread(curr_img_path, mode='RGB').astype(np.uint8)
                    query[i][j] = resize_img(curr_img, size=size, aug=aug)

        return support, query

    def batch_generator(self, label_dim=64, batch_size=64, size=(224, 224), aug=True, mode='train'):
        if mode == 'train':
            all_categories = sorted(list(self.train_label_mapping.keys()))
            img_path_mapping = self.train_img_path_mapping
            label_mapping = self.train_label_mapping
        elif mode == 'val':
            all_categories = sorted(list(self.val_label_mapping.keys()))
            img_path_mapping = self.val_img_path_mapping
            label_mapping = self.val_label_mapping
        elif mode == 'test':
            all_categories = sorted(list(self.test_label_mapping.keys()))
            img_path_mapping = self.test_img_path_mapping
            label_mapping = self.test_label_mapping
        else: 
            raise ValueError('Unknown mode! Please specify mode as either one of train/val/test.')
        
        curr_batch = np.empty((batch_size, size[0], size[1], 3))
        curr_label = np.empty((batch_size, label_dim), dtype=np.uint8)
        img_count = 0
        epoch = 0

        shuffled_idx = list(range(600*len(all_categories)))
        random.shuffle(shuffled_idx)

        while True:
         
            for idx in shuffled_idx:     
                category = all_categories[idx // 600]
                img_idx = idx % 600
                
                #selected_img_path = img_path_mapping[category][img_idx]
                
                label = np.zeros((1, label_dim))
                label[0, label_mapping[category]] = 1

                #s_img = scipy.misc.imread(selected_img_path, mode='RGB').astype(np.uint8)
                s_img = self.train_image[category][img_idx]
                curr_batch[img_count] = resize_img(s_img, size=size, aug=aug)
                curr_label[img_count] = label
                img_count += 1

                if img_count % batch_size == 0 and img_count != 0:        
                    img_count = 0

                    yield curr_batch, curr_label
        
            img_count = 0
            epoch += 1
            print(">>> epoch: {}".format(epoch))
            random.shuffle(shuffled_idx)

    def _load_data(self, categories, path_mapping):
        data_dict = {}
        for category in categories:
            num_img = len(path_mapping[category])
            data_dict[category] = list(range(num_img))
            for i in range(num_img):
                img_path = path_mapping[category][i]
                s_img = scipy.misc.imread(img_path, mode='RGB').astype(np.uint8)
                data_dict[category][i] = s_img
        return data_dict
        
    def batch_generator_load_all(self, label_dim=64, batch_size=64, size=(224, 224), aug=True, mode='train'):
        if mode == 'train':
            all_categories = sorted(list(self.train_label_mapping.keys()))
            img_path_mapping = self.train_img_path_mapping
            label_mapping = self.train_label_mapping
            aug = True
        elif mode == 'val':
            all_categories = sorted(list(self.val_label_mapping.keys()))
            img_path_mapping = self.val_img_path_mapping
            label_mapping = self.val_label_mapping
            aug = False
        elif mode == 'test':
            all_categories = sorted(list(self.test_label_mapping.keys()))
            img_path_mapping = self.test_img_path_mapping
            label_mapping = self.test_label_mapping
            aug = False
        else: 
            raise ValueError('Unknown mode! Please specify mode as either one of train/val/test.')
        
        curr_batch = np.empty((batch_size, size[0], size[1], 3))
        curr_label = np.empty((batch_size, label_dim), dtype=np.uint8)
        img_count = 0
        epoch = 0

        ###
        data_dict = self._load_data(categories=all_categories, path_mapping=img_path_mapping)
        
        shuffled_idx = list(range(600*len(all_categories)))
        random.shuffle(shuffled_idx)

        while True:
        
            for idx in shuffled_idx:     
                category = all_categories[idx // 600]
                img_idx = idx % 600
                
                label = np.zeros((1, label_dim))
                label[0, label_mapping[category]] = 1
                
                s_img = data_dict[category][img_idx]
                curr_batch[img_count] = resize_img(s_img, size=size, aug=aug)
                curr_label[img_count] = label
                img_count += 1

                if img_count % batch_size == 0 and img_count != 0:        
                    img_count = 0

                    yield curr_batch, curr_label
        
            img_count = 0
            epoch += 1
            print(">>> epoch: {}".format(epoch))
            random.shuffle(shuffled_idx)


class MiniImageNet(object):
    def __init__(self, resize=False):
        self.data_path_base = define_dir_by_mac()
        self.data_path = self.data_path_base + 'cross-domain-few-shot/mini-imagenet'
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
        category_list = sorted(list(self.data_dict['train']['class_dict'].keys()))
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
    
    def batch_generator(self, label_dim=64, batch_size=64, size=(224, 224), aug=True, mode='train'):
        # if batch_size % 64 != 0:
        #     print(">>> batch_size = {} should be divisible by 64.".format(batch_size))
        #     return None, None
        
        curr_batch = np.empty((batch_size, 84, 84, 3), dtype=np.uint8)
        curr_label = np.empty((batch_size, label_dim), dtype=np.uint8)
        img_count = 0
        epoch = 0

        all_categories = sorted(list(self.data_dict['train']['class_dict'].keys()))
        shuffled_idx = list(range(600*len(all_categories)))
        random.shuffle(shuffled_idx)

        while True:
         
            for idx in shuffled_idx:     
                category = all_categories[idx // 600]
                img_idx = idx % 600
                if mode == 'train':      
                    selected_img_tag = self.data_dict[mode]['class_dict'][category][img_idx]
                
                label = np.zeros((1, label_dim))
                label[0, self.label_dict[category]] = 1

                s_img = self.data_dict['train']['image_data'][selected_img_tag]
                curr_batch[img_count] = s_img
                curr_label[img_count] = label
                img_count += 1

                if img_count % batch_size == 0 and img_count != 0:        
                    img_batch = resize_batch_img(curr_batch, size=size, aug=aug)
                    img_count = 0

                    yield img_batch, curr_label
        
            img_count = 0
            epoch += 1
            print(">>> epoch: {}".format(epoch))
            random.shuffle(shuffled_idx)


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

def resize_img(img, size, aug): # wei, for process single image       

    if aug:
        resized_img = augmentation(img, size=84)
    else:
        img = center_crop(img, size=size)
        resized_img = ((img / 255.0) - mean) / std
        
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
    height, width = img.shape[0], img.shape[1]
    new_height, new_width = [int(height*1.15), int(width*1.15)] # wei
    
    img = Image.fromarray(img)
    img = img.resize((new_width, new_height))
    
    left = (new_width - width)/2
    top = (new_height - height)/2
    right = (new_width + width)/2
    bottom = (new_height + height)/2

    # Crop the center of the image
    img = img.crop((left, top, right, bottom))
    img = img.resize(size)
    return np.array(img)

def define_dir_by_mac():
    interfaces = netifaces.interfaces()
    
    if 'enp129s0f0' in interfaces:
        default_path = "/data/common/"

    else:
        print("Undefined interface: ", interfaces)
        default_path = "/home/sdc1/dataset/"

    return default_path
