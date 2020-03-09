# io
import os
import cv2
import pickle
import argparse

# computation
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)


class Pacs(object):
    def __init__(self):
        self.domains = ['photo', 'art_painting', 'cartoon', 'sketch']
        self.categories = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.data_path = 'pacs'

    def load_data(self):
        data_dict = {}        
        for domain in self.domains:
            data_dict[domain] = {}
            domain_path = os.path.join(self.data_path, domain)
            for category in self.categories:
                data_dict[domain][category] = []
                category_path = os.path.join(domain_path, category)
                img_files = os.listdir(category_path)
                for img_file in img_files:
                    img = cv2.imread(os.path.join(category_path, img_file))
                    data_dict[domain][category].append(img)
                data_dict[domain][category] = np.stack(data_dict[domain][category])      
        return data_dict


class Cub200(object):
    def __init__(self):
        self.data_path = 'cub200/CUB_200_2011/'
        self.img_path = self._load_img_path()
        self.train_test_split = self._load_train_test_split()
        self.labels = self._load_labels()
        self.class_names = self._load_class_names()
        self.bounding_boxes = self._load_bounding_boxes()

    def _load_img_path(self):
        file_path = os.path.join(self.data_path, 'images.txt')
        img_path = {}
        with open(file_path, 'r') as f:
            for line in f:
                number_vs_path = line.split()
                img_path[number_vs_path[0]] = number_vs_path[1]
        return img_path

    def _load_train_test_split(self):
        file_path = os.path.join(self.data_path, 'train_test_split.txt')
        train_test_split = {}
        with open(file_path, 'r') as f:
            for line in f:
                number_vs_is = line.split()
                train_test_split[number_vs_is[0]] = number_vs_is[1]
        return train_test_split
    
    def _load_labels(self):
        file_path = os.path.join(self.data_path, 'image_class_labels.txt')
        labels = {}
        with open(file_path, 'r') as f:
            for line in f:
                number_vs_label = line.split()
                labels[number_vs_label[0]] = number_vs_label[1]
        return labels

    def _load_class_names(self):
        file_path = os.path.join(self.data_path, 'classes.txt')
        class_names = {}
        with open(file_path, 'r') as f:
            for line in f:
                number_vs_name = line.split()
                class_names[number_vs_name[0]] = number_vs_name[1]
        return class_names

    def _load_bounding_boxes(self):
        file_path = os.path.join(self.data_path, 'bounding_boxes.txt')
        bounding_boxes = {}
        with open(file_path, 'r') as f:
            for line in f:
                number_vs_box = line.split()
                bounding_boxes[number_vs_box[0]] = number_vs_box[1:]
        return bounding_boxes

    def load_data(self):
        data_dict = {}
        data_dict['train'] = {}
        data_dict['test'] = {}
        
        img_index = list(self.img_path.keys())
        for k in img_index:
            img_path = os.path.join(self.data_path, 'images', self.img_path[k])
            img = cv2.imread(img_path)
            class_name = self.class_names[self.labels[k]]
            
            bounding_box = self.bounding_boxes[k]
            x = int(float(bounding_box[0]))
            y = int(float(bounding_box[1]))
            w = int(float(bounding_box[2]))
            h = int(float(bounding_box[3]))
            
            cropped_img = img[y:y+h, x:x+w]

            if self.train_test_split[k] == '1':
                if class_name in data_dict['train'].keys():
                    data_dict['train'][class_name].append(cropped_img)
                else:
                    data_dict['train'][class_name] = []
                    data_dict['train'][class_name].append(cropped_img)
            else:
                if class_name in data_dict['test'].keys():
                    data_dict['test'][class_name].append(cropped_img)
                else:
                    data_dict['test'][class_name] = []
                    data_dict['test'][class_name].append(cropped_img)

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

    f = open('cub_crop.pickle', 'wb')
    pickle.dump(data, f)
    f.close()

