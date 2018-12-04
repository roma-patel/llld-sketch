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
from pretrained_cnns.alexnet import alexnet
from pretrained_cnns.vgg import vgg19

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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize]
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
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

from PIL import Image as Im
def get_photo_sketch_dataset(path, img_size, batch_size, sketch_path):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize]
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = datasets.ImageFolder(path, transform=img_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    classes, class_to_idx, rev_class = get_classes(dataset)
    photos = {class_name: [item.split('.')[0] for item in os.listdir(path + '/' + class_name)] for class_name \
              in classes}
    sketches = {class_name: [item.split('.')[0] for item in os.listdir(sketch_path + '/' + class_name)] for class_name \
              in classes}


    # create tuples = (batch_idx, img1, target_tensor, img_2)
    # first create a list of (img_1, target, img_2) and shuffle
    # then split by batch and create the first tuple
    pairs, data_loader = [], []
    for class_name in photos:
        class_idx = class_to_idx[class_name]
        for photo_idx in photos[class_name]:
            if os.path.isfile(path + '/' + class_name + '/' + photo_idx + '.jpg') is False: continue
            photo = Im.open(path + '/' + class_name + '/' + photo_idx + '.jpg')
            photo_tr = img_transform(photo)
            #### change this!
            #sketch_idxs = [item for item in sketches[class_name]]
            sketch_idxs = [item for item in sketches[class_name] \
                           if item.split('-')[0] == photo_idx]
            for sketch_idx in sketch_idxs:
                sketch = Im.open(sketch_path + '/' + class_name + '/' + sketch_idx + '.png')
                sketch_tr = img_transform(sketch)
                pairs.append((torch.tensor(photo_tr), class_idx, torch.tensor(sketch_tr)))

            #break

    np.random.shuffle(pairs)
    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]

    for batch_idx in range(len(batches)):
        batch = batches[batch_idx]
        targets, photos, sketches = [], [], []
        for item in batch:
            targets.append(item[1])
            photos.append(item[0].data.numpy())
            sketches.append(item[2].data.numpy())

        data_loader.append((batch_idx, (torch.tensor(photos), \
                                               torch.tensor(targets),
                                               torch.tensor(sketches))))

    return dataset, data_loader

def get_classes(dataset):
    classes = [d for d in os.listdir(dataset.root) if os.path.isdir(os.path.join(dataset.root, d))]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    rev_class = {class_to_idx[key]: key for key in class_to_idx.keys()}

    return classes, class_to_idx, rev_class

def get_temp_classes(dataset):
    dirname = os.getcwd() + '/results/files/photo_sketchy/'

    f = open(dirname + 'classes.json', 'r')
    for line in f: classes = json.loads(line)
    #classes = [str(i) for i in range(69)]

    f = open(dirname + 'class_to_idx.json', 'r')
    for line in f: class_to_idx = json.loads(line)
    #class_to_idx = {classes[i]: i for i in range(len(classes))}

    f = open(dirname + 'idx_to_class.json', 'r')
    for line in f:
        temp = json.loads(line)

    idx_to_class = {}
    for key in temp:
        idx_to_class[int(key)] = temp[key]

    #idx_to_class = {class_to_idx[key]: key for key in class_to_idx.keys()}

    return classes, class_to_idx, idx_to_class

def get_word_vectors(wv_path, classes, dim):

    f = open(wv_path, 'r')
    for line in f:
        temp = json.loads(line)

    word_vecs = {}
    for name in classes:
        word_vecs[name] = temp[name]

    return word_vecs

    print("Temporary random vecs!")
    word_vecs = {classes[i]: np.random.rand((dim)) for i in range(len(classes))}
    return word_vecs

def get_wvecs_json(wv_path, classes, dim):

    print("Temporary random vecs!")
    word_vecs = {classes[i]: np.random.rand((dim)) for i in range(len(classes))}
    return word_vecs


    f = open(wv_path, 'r')
    for line in f:
        temp = json.loads(line)

    word_vecs = {}
    for name in classes:
        if name in word_vecs:
            word_vecs[name] = temp[name]

    return word_vecs


def column(matrix, i):
    return [row[i] for row in matrix]

def matrix_to_metrics(confusion_matrix, idx_to_class):
    #tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    diag = np.diag(confusion_matrix)
    overall_acc = sum(diag)/np.sum(confusion_matrix)

    metric_dict = {}
    for class_idx in idx_to_class:
        #print(class_idx)
        tp = confusion_matrix[class_idx][class_idx]
        fp = np.sum(confusion_matrix[class_idx]) - tp
        fn = np.sum(column(confusion_matrix, class_idx)) - tp
        tn = np.sum(confusion_matrix) - tp - fp - fn

        if tp == 0:
            prec, recl, acc = 0.0, 0.0, 0.0
        else:
            prec = max(0.0, tp/(tp+fp))
            recl = max(0,0, tp/(tp+fn))
            acc = (tp+tn)/(tp+fp+fn+fn)
        metric_dict[idx_to_class[class_idx]] = {
            'prec': prec,
            'recl': recl,
            'acc': acc
        }

    return overall_acc, metric_dict

import scipy.stats as stats
def get_spearman(cos_list, sem_list):
    return spearmanr(cos_list, sem_list)[0]

from PIL import Image
import matplotlib.pyplot as plt
def open_quickdraw_file(fname):
    images = np.load(fname)
    for i in range(5, 10):
        img = images[i].reshape((28, 28))

        print(len(img))
        img = np.array(img).astype(np.uint8)
        img_obj = Image.fromarray(img)
        #img_obj = img_obj.resize((128, 128))
        #img_obj.show()

    return []
import PIL.ImageOps
def convert_quickdraw_data(dirpath):
    if os.path.isdir(dirpath + '/images') is False:
        os.mkdir(dirpath+ '/images')

    fnames = os.listdir(dirpath + '/numpy_bitmap')
    print(fnames)
    for fname in fnames:
        #open_quickdraw_file(fname)
        category = fname.split('.')[0]
        if os.path.isdir(dirpath + '/images/' + category + '/') is False:
            os.mkdir(dirpath + '/images/' + category)
        print('Category', fname)
        images = np.load(dirpath + '/numpy_bitmap/' + fname)
        #return
        for i in range(len(images)):
            img = images[i].reshape((28, 28))
            img = np.array(img).astype(np.uint8)
            img_obj = Image.fromarray(img)
            img_obj = PIL.ImageOps.invert(img_obj)
            #img_obj.show()
            img_obj.save(dirpath + '/images/' + category + '/' + str(i) + '.jpg')

import shutil
def add_sketchy_quickdraw():
    # save intersecting categories
    f = open('/home/rpatel59/nlp/llld-sketch/data/files/sketchy_quickdraw_classes.json', 'r')
    for line in f: temp = json.loads(line)
    categories = temp['test'] + temp['train']
    # copy 5 sketchy into test, remaining into train
    sketchy = '/data/nlp/sketchy/figs/256x256/sketch/tx_000000000000/'
    quickdraw = '/data/nlp/quickdraw/images/'

    train_dst = '/data/nlp/gen_sketches/train/'; test_dst = '/data/nlp/gen_sketches/test/'

    # train items

    counts = {}
    for cat in categories:
        if os.path.isdir('/data/nlp/gen_sketches/train/' + cat) is False:
            os.mkdir('/data/nlp/gen_sketches/train/' + cat)
        if os.path.isdir('/data/nlp/gen_sketches/test/' + cat) is False:
            os.mkdir('/data/nlp/gen_sketches/test/' + cat)

        counts[cat] = 0
        fnames = os.listdir(sketchy + cat + '/')
        for fname in fnames[:5]:
            src = sketchy + cat + '/' + fname
            shutil.copy(src, test_dst + cat + '/')
        for fname in fnames[5:]:
            counts[cat] += 1
            src = sketchy + cat + '/' + fname
            shutil.copy(src, train_dst + cat + '/')

        fnames = os.listdir(quickdraw + cat + '/')
        for fname in fnames:
            counts[cat] += 1
            src = quickdraw + cat + '/' + fname
            shutil.copy(src, train_dst + cat + '/')


    f = open('/home/rpatel59/nlp/llld-sketch/data/files/counts-gen.json', 'w+')
    f.write(json.dumps(counts))

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

def get_attrs(class_to_idx):
    attr_dict = {}
    f = open(os.getcwd() + '/data/files/attr.json', 'r')
    for line in f:
        attrs = json.loads(line)

    f = open(os.getcwd() + '/data/files/attrs.tsv', 'r')
    attr_names = [line.strip() for line in f.readlines()]

    existing_attrs = []

    for cat in class_to_idx.keys():
        existing_attrs.extend([attr_names[i] for i in range(len(attr_names)) if attrs[cat][i] > 0])

    existing_attrs = [attr for attr in existing_attrs if '-' not in attr]
    existing_attrs = sorted(list(set(existing_attrs)))

    print(existing_attrs)
    for attr in attrs:
        if attr in class_to_idx.keys():
            temp = attrs[attr]
            attr_dict[attr] = [temp[i] for i in range(len(temp)) if attr_names[i] in existing_attrs]
            
    return attr_dict, len(existing_attrs)

def move_data():
    dirpath = '/data/nlp/sketchy_splits/'
    folders = ['sketch/', 'sketch_3/', 'photo/']

    f = open(os.getcwd() + '/data/files/sketchy_classes.json', 'r')
    for line in f: classes = json.loads(line)

    train, test = classes['train'], classes['test']

    for folder in folders:
        for class_name in test:
            src_dir = dirpath + folder + 'test/' + class_name + '/'
            dest_dir = dirpath + folder + 'train/' + class_name + '/'

            fnames = os.listdir(src_dir)
            test_files = fnames[:5]
            train_files = fnames[5:]
            for file in train_files:
                print(file)
                print(src_dir); print(dest_dir); print('\n')
                shutil.move(src_dir + file, dest_dir)
        
def get_img_rep(model, image):
    print('Image!')
    print(image)
    rep = model.forward(image)
    print(rep)
    print('\n\n\n')
    return rep

if __name__ == '__main__':
    #convert_quickdraw_data('/data/nlp/quickdraw/')

    img_size, batch_size = 256, 64
    datapath = '/Users/romapatel/github/llld-sketch/data/temp/animals/natural/'
    dataset, data_loader = get_dataset(datapath, img_size, \
            batch_size)

    model = vgg19(pretrained=True)
    #model = alexnet(pretrained=True)

    for batch_idx, (img, target_tensor) in enumerate(data_loader):
        get_img_rep(model, img)
    #add_sketchy_quickdraw()
    #move_data()

    #convert_quickdraw_data('/Users/romapatel/Desktop/quickdraw/')
    #open_quickdraw_file('/Users/romapatel/Desktop/quickdraw/numpy_bitmap/zebra.npy')
    #interview()
    args = 'sketch_temp sketch /Users/romapatel/github/sketch-attr/'.split()
    #log.basicConfig(filename=os.getcwd() + '/logs/sketch_' + str(sketch_idx) + '.log',level=log.DEBUG)



    



    
