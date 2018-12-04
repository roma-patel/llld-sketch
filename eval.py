import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import itertools, json, os, sys, random, argparse
from torchvision.utils import save_image
import logging as log
from src import modules, utils
from src.modules import *
from src.utils import *
import scipy.stats as stats
import operator
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from scipy.stats import pearsonr, spearmanr
from pretrained_cnns.alexnet import alexnet
from pretrained_cnns.vgg import vgg19
from pretrained_cnns.vgg import vgg11
from pretrained_cnns.resnet import resnet101

def handle_args(args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', '-c', type=str, nargs='+',
            help='Config file for model parameters.')
    parser.add_argument('--overrides', '-o', type=str, default=None,
            help='Any overrides to default configurations.')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
            help='Batch size (elements per batch) during training.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3,
            help='')

    parser.add_argument('--split', '-s', type=float, default=0.2,
            help='Train/test proportion of all data.')
    parser.add_argument('--epochs', '-epochs', type=int, default=1,
            help='Number of epochs.')
    parser.add_argument('--emb_dim', '-emb_dim', type=int, default=100,
            help='Embedding dimension size for neural modules.')
    parser.add_argument('--hid_dim', '-hid_dim', type=str, default=100,
            help='Hidden layer size for neural modules.')
    parser.add_argument('--run_name', '-run_name', type=str, \
            default='sketch_sketchy_temp', help='Name after which \
            files/results generated will be saved.')
    parser.add_argument('--img_size', '-img_size', type=int, default=256,
            help='')
    parser.add_argument('--datadir', '-datadir', type=str, \
                        default= '/Users/romapatel/github/llld-sketch/data/temp/animals/sketch/',
            help='')
    parser.add_argument('--eval', '-eval', type=str, default='True',
            help='')
    parser.add_argument('--test_datadir', '-test_datadir', type=str, \
                        default= '/Users/romapatel/github/llld-sketch/data/temp/animals/natural/',
            help='')
    parser.add_argument('--fin_run', '-fin_run', type=str, default='only-reconstr',
            help='')


    return parser.parse_args(args)


# reconstruct only
def model_1(args):
    if os.path.isdir(os.getcwd() + '/results/images/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/images/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/history/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/history/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/files/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/files/' + args.run_name)

    datapath = args.datadir
    args.img_size = 224

    dataset, data_loader = utils.get_dataset(datapath, args.img_size, \
            args.batch_size)
    classes, class_to_idx, idx_to_class = utils.get_classes(dataset)
    word_dim = 300
    label_dim = len(classes)


    model = BimodalDAEImage(300, 2048, n_classes=len(classes))
    cnn = resnet101(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))

    word_vecs = utils.get_wvecs_json(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)
    #word_vecs = utils.get_word_vectors(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)

    encoding_dict = {}
    with torch.no_grad():
        for batch_idx, (img, target_tensor) in enumerate(data_loader):

            target_idxs = target_tensor.data.numpy().tolist()
            target_names = [idx_to_class[idx] for idx in target_idxs]
            target_labels = torch.tensor([[1 if i == idx else 0 for i in \
                    range(len(classes))] for idx in target_idxs], \
                    dtype=torch.long)

            # previously target dist reps
            target_textual = torch.tensor([word_vecs[name] for name in target_names], \
                                            dtype=torch.float32)

            target_visual = torch.tensor(
                [cnn.forward(
                    img[idx].reshape(1, 3, args.img_size, args.img_size)).data.numpy() for idx in range(len(target_idxs))], dtype=torch.float32
            )

            n_samples = len(target_idxs)

            img_reconstr, text_reconstr, hidden = model.forward(target_visual, \
                                                              target_textual)

            print('Hidden', hidden.size())
            #preds = softmax(hidden)
            reps = hidden.data.numpy()
            for idx in range(len(reps)):
                target = target_names[idx]
                print(target)
                if target not in encoding_dict.keys():
                    encoding_dict[target] = []
                #val = reps[idx].view(1, -1)
                encoding_dict[target].append(list(reps[idx].tolist()))


    f = open(os.getcwd() + '/results/files/' + args.run_name + '/encoding_dict.json', 'w+')
    f.write(json.dumps(encoding_dict))
    print('Eval done!')


def model_2(args):
    return

def model_3(args):

    if os.path.isdir(os.getcwd() + '/results/images/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/images/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/history/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/history/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/files/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/files/' + args.run_name)

    datapath = args.datadir
    args.img_size = 224

    dataset, data_loader = utils.get_dataset(datapath, args.img_size, \
            args.batch_size)
    classes, class_to_idx, idx_to_class = utils.get_classes(dataset)
    word_dim = 300
    label_dim = len(classes)
    attr_dict, n_attrs = get_attrs(class_to_idx)


    model = BimodalDAEAttr(300, 2048, n_attrs, n_classes=len(classes))
    cnn = resnet101(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))

    word_vecs = utils.get_wvecs_json(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)
    #word_vecs = utils.get_word_vectors(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)

    encoding_dict = {}
    with torch.no_grad():
        for batch_idx, (img, target_tensor) in enumerate(data_loader):

            target_idxs = target_tensor.data.numpy().tolist()
            target_names = [idx_to_class[idx] for idx in target_idxs]
            target_labels = torch.tensor([[1 if i == idx else 0 for i in \
                    range(len(classes))] for idx in target_idxs], \
                    dtype=torch.long)
            target_attrs = torch.tensor([attr_dict[idx_to_class[idx]] for idx in \
                    target_idxs], dtype=torch.float32)
            # previously target dist reps
            target_textual = torch.tensor([word_vecs[name] for name in target_names], \
                                            dtype=torch.float32)

            target_visual = torch.tensor(
                [cnn.forward(
                    img[idx].reshape(1, 3, args.img_size, args.img_size)).data.numpy() for idx in range(len(target_idxs))], dtype=torch.float32
            )

            n_samples = len(target_idxs)

            img_reconstr, text_reconstr, hidden = model.forward(target_visual, \
                                                              target_textual)

            print('Hidden', hidden.size())
            #preds = softmax(hidden)
            reps = hidden.data.numpy()
            for idx in range(len(reps)):
                target = target_names[idx]
                print(target)
                if target not in encoding_dict.keys():
                    encoding_dict[target] = []
                #val = reps[idx].view(1, -1)
                encoding_dict[target].append(list(reps[idx].tolist()))


    f = open(os.getcwd() + '/results/files/' + args.run_name + '/encoding_dict.json', 'w+')
    f.write(json.dumps(encoding_dict))
    print('Eval done!')


from sklearn.decomposition import PCA
def get_corr(args):
    #classes, class_to_idx, idx_to_class = utils.get_classes(dataset)

    f = open(os.getcwd() + '/results/files/' + args.run_name + '/encoding_dict.json', 'r')
    for line in f:
        reps = json.loads(line)


    f = open(os.getcwd() + '/data/files/sketchy_classes.json', 'r')
    for line in f: class_splits = json.loads(line)

    f = open(os.getcwd() + '/data/files/class_to_idx.json', 'r')
    for line in f: class_to_idx = json.loads(line)
    classes = class_splits['train']

    #attr_dict, n_attrs = get_attrs(class_to_idx)


    for name in reps:
        print(name)
        print(len(reps[name]))
    return
    '''
    f = open('/Users/romapatel/Desktop/avg_vgg128_nouns.csv', 'r')
    lines = f.readlines()

    vgg_dict = {}
    for line in lines:
        items = line.strip().split(',')
        #print(items[0])
        if items[0] in classes:
            vgg_dict[items[0]] = [float(item) for item in items[1:]]

    '''


    # finally run this using the function in utils

    print(len(classes))

    f = open(os.getcwd() + '/data/files/sem-vis-sketchy.tsv', 'r')
    lines = [line.strip().split('\t') for line in f.readlines()]
    # evaluate only the first


    class_rep_dict, sims, true = {}, [], []
    for key in reps:
        val = class_to_idx[int(key)]
        if val not in classes: continue
        class_name = classes[int(key)]
        # evaluate only the first
        class_rep_dict[class_name] = reps[key]



    encoding_dict = class_rep_dict
    for key in encoding_dict:
        val = class_to_idx[key]
        if key not in encoding_dict.keys(): continue
        print(len(encoding_dict[key]))
        #print(encoding_dict[key])
        sims = []
        for rep1 in encoding_dict[key]:
            for rep2 in encoding_dict[key]:
                sims.append(cos_sim(np.array(rep1).reshape(1, -1), \
                                    np.array(rep2).reshape(1, -1)))
        #print(sims)
        print(np.mean(sims))
    return

    '''
    f = open(os.getcwd() + '/data/files/wvecs.json', 'r')
    for line in f: wvecs = json.loads(line)
    print(wvecs)
    '''

    #class_rep_dict = attr_dict
    for line in lines:
        word1, word2 = line[0], line[1]
        if word1 not in class_rep_dict.keys(): continue
        if word2 not in class_rep_dict.keys(): continue

        print(len(class_rep_dict[word1]))
        rep1 = np.array(class_rep_dict[word1]).reshape(1, -1)
        rep2 = np.array(class_rep_dict[word2]).reshape(1, -1)

        sim = cos_sim(rep1, rep2)[0][0]
        sims.append(sim)
        true.append(float(line[2]))
        s = word1 + '-' + word2
        print(s)
        print(cos_sim(rep1, rep2))




    pearson = pearsonr(sims, true)
    spearman = spearmanr(sims, true)
    print(pearson)
    print(spearman)


if __name__ == '__main__':

    args = sys.argv[1:]
    handled_args = handle_args(args)
    if os.path.isdir(os.getcwd() + '/logs/') is False:
        os.mkdir(os.getcwd() + '/logs/')
    fname = os.getcwd() + '/logs/' + handled_args.run_name + '.log'
    log.basicConfig(filename = fname, format='%(asctime)s: %(message)s', \
                    datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

    #handled_args.run_name = 'photo-attr-only'
    #handled_args.fin_run = 'only-attr'


    handled_args.fin_run = 'photo-reconstr'


    get_corr(handled_args)

    '''
    if handled_args.fin_run == 'photo-reconstr':
        model_1(handled_args)
    elif handled_args.fin_run == 'sketch-reconstr':
        model_2(handled_args)
    elif handled_args.fin_run == 'attr-reconstr':
        model_3(handled_args)
    elif handled_args.fin_run == 'unimodal-photo':
        model_4(handled_args)
    elif handled_args.fin_run == 'unimodal-sketch':
        model_5(handled_args)
    '''
