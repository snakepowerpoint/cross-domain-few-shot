# io
import os
import cv2

# computation
import numpy as np
import random


class Pacs(object):
    def __init__(self):
        self.domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        self.categories = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.data_path = 'data/pacs'
        
    def get_task(self, domain, categories, n_shot, n_query):
        n_way = len(categories)
        domain_path = os.path.join(self.data_path, domain)
        task = {}
        task['support'] = []
        task['query'] = []
        for category in categories:
            category_path = os.path.join(domain_path, category)
            img_files = os.listdir(category_path)
            img_files = random.sample(img_files, k=n_shot + n_query)
            for img_file in img_files[:n_shot]:
                img = cv2.imread(os.path.join(category_path, img_file))
                task['support'].append(img)
            for img_file in img_files[n_shot:]:
                img = cv2.imread(os.path.join(category_path, img_file))
                task['query'].append(img)

        task['support'] = np.stack(task['support'])
        task['query'] = np.stack(task['query'])

        support_shape = np.concatenate(([n_way, n_shot], task['support'].shape[1:]))
        query_shape = np.concatenate(([n_way, n_query], task['query'].shape[1:]))
        
        task['support'] = np.reshape(task['support'], support_shape)
        task['query'] = np.reshape(task['query'], query_shape)
        return task
