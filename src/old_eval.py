import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.optim as optim
import numpy as np
import itertools, json, os, sys, operator, math
#from PIL import Image
#import matplotlib.pyplot as plt
from torchvision.utils import save_image
import logging as log
from pandas import DataFrame as df
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from scipy.stats import pearsonr, spearmanr

class ImageEncoder:
    def __init__(self, input_id, hidden_dim=1000):
        self.input_id = input_id

    def forward(self):
        rep = torch.tensor([i/1000.0 for i in range(1000)])
        return rep
    
class TextEncoder:
    def __init__(self, input_id, hidden_dim=1000):
        self.input_id = input_id
        
    def forward(self):
        rep = torch.tensor([i/1000.0 for i in range(1000)])
        return rep

class AutoEncoder(nn.Module):
    def __init__(self, img_idx, width, height, attr_dict, label_dim, attr_dim):
        super(AutoEncoder, self).__init__()
        self.img_idx = img_idx
        self.attrs = attr_dict
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            #nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            #nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )
        self.flattened_dim = 3528
        self.label_dim = label_dim
        self.attr_transform = nn.Linear(self.flattened_dim, attr_dim)
        #self.label_transform = nn.Linear(self.flattened_dim, label_dim)
        self.label_transform = nn.Linear(attr_dim, label_dim)

    def __str__(self):
        return str((self.img_idx))
    
     
    def bottleneck(self, rep):
        states = self.goal_states()        
        transform = nn.Linear(self.hidden_dim, len(states))
        rep = transform(rep)
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    def forward(self, input):
        # encode
        hidden = self.encoder(input)
        # classify hidden
        attr_weights, attr = self.attributes(hidden)
        #labels = self.labels(hidden)
        labels = self.labels(attr_weights)

        # decode
        reconstr = self.decoder(hidden)

        return reconstr, attr, labels

    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)

        return indices[-k:]
    
    def attributes(self, rep):
        rep = rep.view(rep.size(0), -1)
        attr_weights = self.attr_transform(rep)
        sigmoid = nn.Sigmoid()
        attr_dist = sigmoid(attr_weights)

        return attr_weights, attr_dist

    def labels(self, rep):
        rep = rep.view(rep.size(0), -1)
        label_weights = self.label_transform(rep)
        softmax = nn.Softmax(dim=1)
        label_dist = softmax(label_weights)

        #labels = torch.argmax(label_dist, dim=1)
        return label_dist


def get_weights(model_name, path):
    dirpath = os.getcwd() + '/model_states/'
    if os.path.isdir(os.getcwd() + '/results/reps/') is False:
        os.mkdir(os.getcwd() + '/results/reps/')


    img_size, img_type, attr_dict = 256, 'sketch', {}
    batch_size = 128; 
    img_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )
    
    print(path)
    dataset = datasets.ImageFolder(path, transform=img_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    classes = [d for d in os.listdir(dataset.root) if os.path.isdir(os.path.join(dataset.root, d))]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    rev_class = {class_to_idx[key]: key for key in class_to_idx.keys()}
    attr_dict = get_attrs(class_to_idx)
    n_attrs = len(attr_dict[list(class_to_idx.keys())[0]])
    n_classes = len(class_to_idx)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Loaded test data!')
    model = AutoEncoder(img_idx='cat_0', width=img_size, height=img_size, attr_dict=attr_dict, label_dim=n_classes, attr_dim = n_attrs)

    checkpoint = torch.load(os.getcwd() + '/model_states/' + model_name)
    model.load_state_dict(checkpoint['state_dict'])

    identity = torch.eye(n_attrs)
    weights = model.label_transform(identity)
    weights = weights.data.numpy().tolist()

    print(weights)
    
    dict = {'weights': weights, 'classes': class_to_idx}
    f = open(os.getcwd() + '/results/reps/' + model_name + '.json', 'w+')
    f.write(json.dumps(dict))
    print('Done!')

def print_weights(model_name, path):
    '''
    dirpath = os.getcwd() + '/results/reps/'
    f = open(dirpath + model_name + '.json', 'r')
    for line in f:
        dict = json.loads(line)
    '''
    dirpath = os.getcwd() + '/results/reps/'
    f = open(dirpath + model_name + '.json', 'r')
    for line in f: temp = json.loads(line)

    matrix = temp['weights']
    classes = temp['classes']
    sorted_x = sorted(classes.items(), key=operator.itemgetter(1))
    columns = [item[0] for item in sorted_x]
    print(sorted_x)
    dataframe = df(matrix, columns=columns) 
    print(dataframe)

    f = open(os.getcwd() + '/data/files/sem-vis.txt', 'r')
    lines = [line.strip().split('\t') for line in f.readlines()]
    true = []
    for line in lines:
        items = line[0].split('#')
        if items[0] in columns and items[1] in columns:
            true.append(line)
            print(line)
        else:
            print('Not')

    print(len(true))
    if 'tu_int' in model_name:
        f = open(os.getcwd() + '/data/files/sem-vis-tu_int.tsv', 'w+')
    else:
        f = open(os.getcwd() + '/data/files/sem-vis-sketchy.tsv', 'w+')

    for line in true:
        words = line[0].split('#')
        f.write(words[0] + '\t' + words[1] + '\t' + line[1] + '\t' + line[2] + '\n')
    return None

def get_corr(model_name, path):
    dirpath = os.getcwd() + '/results/reps/'
    f = open(dirpath + model_name + '.json', 'r')
    for line in f: temp = json.loads(line)

    matrix = temp['weights']
    classes = temp['classes']
    sorted_x = sorted(classes.items(), key=operator.itemgetter(1))
    columns = [item[0] for item in sorted_x]
    print(sorted_x)
    dataframe = df(matrix, columns=columns) 
    #print(dataframe)

    if 'tu_int' in model_name:
        f = open(os.getcwd() + '/data/files/sem-vis-tu_int.tsv', 'r')
    else:
        f = open(os.getcwd() + '/data/files/sem-vis-sketchy.tsv', 'r')

    lines = [line.strip().split('\t') for line in f.readlines()]
    
    true = [float(line[2]) for line in lines]
    reps, sims = {}, []
    for column in columns:
        reps[column] = np.array([item if math.isnan(item) is False else 0 for item in dataframe[column].values])

    
    for line in lines:
        word1, word2 = line[0], line[1]
        sims.append(cos_sim([reps[word1]], [reps[word2]])[0][0])

    f = open(os.getcwd() + '/results/reps/' + model_name + '_corr.txt', 'w+')
    f.write('Pearson:'); f.write(str(pearsonr(sims, true)))
    f.write('\nSpearman:'); f.write(str(spearmanr(sims, true)))


    '''
    print(sims)
    '''
    print('\n\n')
    print('Pearson')
    print(pearsonr(sims, true))

    print('\nSpearman')
    print(spearmanr(sims, true))

def print_attrs(model_name):
    '''
    f = open(os.getcwd() + '/data/files/attrs.tsv', 'r')
    attrs = [line.strip() for line in f.readlines()]

    f = open(os.getcwd() + '/data/files/attr.json', 'r')
    for line in f:
        attr_dict = json.loads(line)

    existing_attrs = []
    f = open(os.getcwd() + '/data/files/tu_classes.json', 'r')
    for line in f: sketchy = json.loads(line)

    cats = sketchy['train'] + sketchy['test']
    for cat in cats:
        existing_attrs.extend([attrs[i] for i in range(len(attrs)) if attr_dict[cat][i] > 0])


    print(set(existing_attrs))
    print(len(set(existing_attrs)))
    return
    '''
    dirpath = os.getcwd() + '/results/reps/'

    dirpath = '/Users/romapatel/github/sketch-attr/' + 'results/reps/'
    f = open(dirpath + model_name + '.json', 'r')
    for line in f: temp = json.loads(line)

    matrix = temp['weights']
    classes = temp['classes']

    #f = open(os.getcwd() + '/data/files/attrs.tsv', 'r')
    f = open('/Users/romapatel/github/sketch-attr/' + '/data/files/attrs.tsv', 'r')

    attrs = [line.strip() for line in f.readlines()]
    sorted_x = sorted(classes.items(), key=operator.itemgetter(1))
    columns = [item[0] for item in sorted_x]
    print(sorted_x); print()
    dataframe = df(matrix, columns=columns) 

    # for category print attrs
    for column in columns:
        vec = dataframe[column]
        vals = {attrs[i]: vec[i] for i in range(len(vec))}
        print(column)
        sorted_x = sorted(vals.items(), key=operator.itemgetter(1), reverse=True)
        print(sorted_x[:10])




def to_img(x, img_size):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    #x = x.view(x.size(0), 1, 28, 28)
    x = x.view(x.size(0), 3, img_size, img_size)

    return x


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
    existing_attrs = list(set(existing_attrs))
    for attr in attrs:
        if attr in class_to_idx.keys():
            temp = attrs[attr]
            attr_dict[attr] = [temp[i] for i in range(len(temp)) if attr_names[i] in existing_attrs]
    return attr_dict


# takes a attr vector and returns indices above threshold
def get_pred_attrs(attr_dist):
    threshold = 0.7
    #values, indices = attr_dist.sort(dim=0) 
    idxs = []
    for row in attr_dist:
        row_idxs = (row>=threshold).nonzero()
        row_idxs = row_idxs.data.numpy().tolist()
        idxs.append([item[0] for item in row_idxs])

    return idxs

def get_prec_recl_acc(pred_labels, target_labels):
    #pred_labels = [1, 0, 1, 0]
    #target_labels = [1, 1, 0, 0] 
    true_pos = [1 if pred_labels[i] == target_labels[i] and pred_labels[i] == 1 else 0 for i in range(len(pred_labels))]
    true_neg = [1 if pred_labels[i] == target_labels[i] and pred_labels[i] == 0 else 0 for i in range(len(pred_labels))]
    false_pos = [1 if pred_labels[i] == 1 and target_labels[i] == 0 else 0 for i in range(len(pred_labels))]
    false_neg = [1 if pred_labels[i] == 0 and target_labels[i] == 1 else 0 for i in range(len(pred_labels))]

    acc = 100*float(sum(true_pos) + sum(true_neg))/ len(pred_labels)
    if sum(true_pos) == 0: 
        prec = 0.0; recl = 0.0
    else:
        prec = 100*float(sum(true_pos))/ (sum(true_pos) + sum(false_pos))
        recl = 100*float(sum(true_pos))/ (sum(true_pos) + sum(false_neg))
    return prec, recl, acc


def get_metrics(pred_labels, target_labels, pred_attrs, target_attrs):
    metrics = {'labels': {'acc': 0, 'prec': 0, 'recl': 0}, 'attr': {'acc': [], 'prec': [], 'recl': []}}

    pred_labels = pred_labels.data.numpy().tolist()

    true_pos = [1 if pred_labels[i] == target_labels[i] else 0 for i in range(len(pred_labels))]

    metrics['labels']['acc'] = 100*float(sum(true_pos))/len(true_pos)

    target_attrs = get_pred_attrs(target_attrs)

    pred_vec = [[1 if i in item else 0 for i in range(104)] for item in pred_attrs]
    true_vec = [[1 if i in item else 0 for i in range(104)] for item in target_attrs]

    for i in range(len(pred_vec)):
        prec, recl, acc = get_prec_recl_acc(pred_vec[i], true_vec[i])
        metrics['attr']['acc'].append(acc)
        metrics['attr']['prec'].append(prec)
        metrics['attr']['recl'].append(recl)

    for key in metrics['attr']:
        mean = np.mean(metrics['attr'][key])
        metrics['attr'][key] = mean


    return metrics

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


def update_label_matrix(matrix, pred_labels, target_labels):
    pred_labels = pred_labels.data.numpy().tolist()
    target_labels = torch.argmax(target_labels, dim=1)
    target_labels = target_labels.data.numpy().tolist()

    print(pred_labels); print(target_labels); print()
    return matrix


def test(name, path):
    img_size, img_type, attr_dict = 256, 'sketch', {}
    test_name = name + '_test'
    
    num_epochs = 100; batch_size = 128; learning_rate = 1e-3
    img_transform = transforms.Compose(
            [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )
    
    dataset = datasets.ImageFolder(path, transform=img_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    classes = [d for d in os.listdir(dataset.root) if os.path.isdir(os.path.join(dataset.root, d))]
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    rev_class = {class_to_idx[key]: key for key in class_to_idx.keys()}

    attr_dict = get_attrs(class_to_idx)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Loaded test data!')
    print(len(data_loader))
    criterion = nn.MSELoss()
    attr_criterion = nn.L1Loss()
    model = AutoEncoder(img_idx='cat_0', width=img_size, height=img_size, attr_dict=attr_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)

    checkpoint = torch.load(os.getcwd() + '/model_states/' + name)
    model.load_state_dict(checkpoint['state_dict'])

    #accs = {'reconstr': [], 'attr': [], 'labels': []}

    f = open(os.getcwd() + '/results/files/' + test_name + '.txt', 'w+')
    print('Loaded model!')

    n_classes = len(class_to_idx)
    print("Number of classes: " + str(n_classes))
    matrix = np.zeros((n_classes, n_classes))
    if os.path.isdir(os.getcwd() + '/results/images/' + test_name) is False:
        os.mkdir(os.getcwd() + '/results/images/' + test_name)
    with torch.no_grad():
        for batch_idx, (img, target_tensor) in enumerate(data_loader):
            target_idxs = target_tensor.data.numpy().tolist()
            target_names = [rev_class[idx] for idx in target_idxs]
            target_attrs = torch.tensor([attr_dict[item] for item in target_names], dtype=torch.float32) 
            target_labels = torch.tensor([[1 if i == idx else 0 for i in range(104)] for idx in target_idxs]
, dtype=torch.float32) 
            
            reconstr, attr_dist, label_dist = model.forward(img)
            pred_labels = torch.argmax(label_dist, dim=1)
            pred_attrs = get_pred_attrs(attr_dist)
            metrics = get_metrics(pred_labels, target_idxs, pred_attrs, target_attrs)
            #print(metrics)
            matrix = update_label_matrix(matrix, pred_labels, target_labels)
            s = pretty_print(metrics, name)
            print(s)
            f.write('\nEpoch: ' + str(batch_idx) + ', Batch: ' + str(batch_idx) + '\n' + s + '\n')
            
            '''
            #if epoch%10 == 0:
            reconstr_img = to_img(reconstr.data, img_size)
            save_image(reconstr_img, os.getcwd() + '/results/images/' + test_name + '/epoch_' + str(batch_idx) + '.png')
            '''


if __name__ == '__main__':
    '''
    model_name = sys.argv[1]
    path = sys.argv[2]
    '''
    model_name = 'photo_sketchy_encoder'
    path = ''
    #get_weights(model_name, path)
    #print_weights(model_name, path)
    #get_corr(model_name, path)
    print_attrs(model_name)
    #test(model_name, path)
    


    



    
