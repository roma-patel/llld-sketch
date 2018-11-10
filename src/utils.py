import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.optim as optim
import numpy as np
import itertools, json, os, sys, shutil
#from PIL import Image
from torchvision.utils import save_image
import logging as log
import itertools

def to_img(x, img_size):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, img_size, img_size)
    return x

# function that takes in the matrix and updates confusion matrix
def update_label_matrix(matrix, labels, target_idxs):
    #for (pred_idx, true_idx) in itertools.product(labels, target_idxs):
    for i in range(len(labels)):
        pred_idx, true_idx = labels[i], target_idxs[i]
        matrix[pred_idx.data.numpy().tolist()][true_idx] += 1
    return matrix

def update_attr_matrix(matrix, labels, target_idxs):
    for (pred_idx, true_idx) in itertools.product(labels, target_idxs):
        matrix[pred_idx][true_idx] += 1
    return matrix

def get_dataset(path, img_size, batch_size):
    img_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )
    dataset = datasets.ImageFolder(path, transform=img_transform)

    '''
    counts = {i: 0 for i in range(124)}
    n_samples = 2
    print(counts)
    print(dataset)
    for item in dataset:
        c = item[1]
        if counts[c] >= n_samples: continue
        counts[c] += 1
    print(counts)
    '''
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, data_loader

def get_classes(dataset):
    classes = [d for d in os.listdir(dataset.root) if os.path.isdir(os.path.join(dataset.root, d))]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    rev_class = {class_to_idx[key]: key for key in class_to_idx.keys()}

    return classes, class_to_idx, rev_class

def column(matrix, i):
    return [row[i] for row in matrix]

def matrix_to_metrics(confusion_matrix, idx_to_class):
    #tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    diag = np.diag(confusion_matrix)
    overall_acc = sum(diag)/np.sum(confusion_matrix)

    metric_dict = {}
    for class_idx in idx_to_class:
        tp = confusion_matrix[class_idx][class_idx]
        fp = np.sum(confusion_matrix[class_idx]) - tp
        fn = np.sum(column(confusion_matrix, class_idx)) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn
        metric_dict[idx_to_class[class_idx]] = {
            'prec': max(0.0, tp/(tp+fp)),
            'recl': max(0,0, tp/(tp+fn)),
            'acc': (tp+tn)/(tp+fp+fn+fn)
        }

    return overall_acc, metric_dict

def add_items(items, name, rank):
    f = open(os.getcwd() + '/results/files/' + name + '/matrix.json', 'r')
    for line in f:
        temp = json.loads(line)

    f = open(os.getcwd() + '/results/files/' + name + '/idx_to_class.json', 'r')
    for line in f:
        idx_to_class = json.loads(line)

    categories = [idx_to_class[idx] for idx in list(idx_to_class.keys())]

    for i in range(len(categories)):
        row = temp['matrix'][i]
        dtype = [('name', 'S10'), ('count', int)]
        vals = [(categories[i], row[i]) for i in range(len(row))]
        a = np.array(vals, dtype=dtype)
        a = np.sort(a, order='count')
        a = (a[::-1][:rank])
        s = name.split('_')[0] + '\n' + categories[i] + ' :: ' + ' '.join(str(item[0].strip()) for item in a)
        items[i].append(s)
    return items

def get_acc_items(items):
    accs = []
    for item in items:
        vals = item[0].strip().split(' :: ')
        cat = vals[1]
        preds = vals[-1].strip().split(') (')
        preds = [pred.split('\'')[1] for pred in preds]
        if cat in preds: accs.append(1)

    print(float(sum(accs))/len(items))




def pretty_print(metrics, name):
    cols = ['label_acc', 'attr_acc', 'attr_prec', 'attr_recl']
    s = '-' + '\t'
    for col in cols: s += col + '\t'
    s += '\n'
    s += name + '\t'

    s += str(round(metrics['labels']['acc'], 2)) + '\t'
    s += str(round(metrics['attr']['acc'], 2)) + '\t'
    s += str(round(metrics['attr']['prec'], 2)) + '\t'
    s += str(round(metrics['attr']['recl'], 2)) + '\t'

    return s


def interview():

    def search_matrix():
        a = np.zeros((3, 3)); a[0][2] = 3
        val = 3
        start_row, end_row, start_col, end_col = 0, len(a), 0, len(a)
        while True:
            mid_row = (start_row+end_row)/2
            mid_col = (start_col+end_col)/2

            current = a[mid_row][mid_col]
            if val == current:
                flag = True
                break
            elif val < current:
                end_row = mid_row
            else:
                start_row = mid_row

    search_matrix()

if __name__ == '__main__':
    interview()
    args = 'sketch_temp sketch /Users/romapatel/github/sketch-attr/'.split()
    #log.basicConfig(filename=os.getcwd() + '/logs/sketch_' + str(sketch_idx) + '.log',level=log.DEBUG)



    



    
