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


# need to correct
class Cub(object):
    def __init__(self, img_size, interpolation=cv2.INTER_LINEAR):
        self.data_path = 'data/'
        self.data_dict = self._load_data(img_size=img_size, interpolation=interpolation)
    
    def _load_data(self, img_size, interpolation):
        data_path = os.path.join(self.data_path, 'cub_crop.pickle')
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
            for dataset, class_dict in data_dict.items():
                for class_name, image_list in class_dict.items():
                    data_dict[dataset][class_name] = []
                    for image in image_list:
                        resized_img = cv2.resize(image, (img_size, img_size), interpolation)
                        data_dict[dataset][class_name].append(resized_img)
                    data_dict[dataset][class_name] = np.stack(data_dict[dataset][class_name])
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


