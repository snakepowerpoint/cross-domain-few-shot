# io
import os
import pickle
import cv2

# computation
import numpy as np
import random


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
    def __init__(self):
        self.data_path = 'data/'
        self.data_dict = self._load_data()

    def _load_data(self):
        data_path = os.path.join(self.data_path, 'cub_crop.pickle')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    def get_task(self, n_way=5, n_shot=5, n_query=15, mode='test'):
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
    def __init__(self):
        self.data_path = 'data/mini-imagenet'
        self.data_dict = self._load_data()

    def _load_data(self):
        train_data_path = os.path.join(self.data_path, 'mini-imagenet-cache-train.pkl')
        test_data_path = os.path.join(self.data_path, 'mini-imagenet-cache-test.pkl')
        data_dict = {}
        with open(train_data_path, 'rb') as f:
            data_dict['train'] = pickle.load(f)
        with open(test_data_path, 'rb') as f:
            data_dict['test'] = pickle.load(f)
        
        data_dict = self._normalized(data_dict)
        return data_dict
    
    def _normalized(self, data_dict):
        data_dict['train']['image_data'] = data_dict['train']['image_data'] / 255.0 
        data_dict['test']['image_data'] = data_dict['test']['image_data'] / 255.0
        return data_dict

    def get_task(self, n_way=5, n_shot=5, n_query=15, mode='train'):
        support = []
        query = []
        selected_categories = random.sample(list(self.data_dict[mode]['class_dict'].keys()), k=n_way)
        for category in selected_categories:
            selected_imgs = random.sample(
                self.data_dict[mode]['class_dict'][category], k=n_shot+n_query)
            support.append(self.data_dict[mode]['image_data'][selected_imgs[:n_shot]])
            query.append(self.data_dict[mode]['image_data'][selected_imgs[n_shot:]])

        support = np.stack(support)
        query = np.stack(query)
        return support, query
