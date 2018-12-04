import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.optim as optim
import numpy as np
import itertools, json, os, sys, random, argparse
from torchvision.utils import save_image
import logging as log
from src import modules, utils
from src.modules import *
from src.utils import *
import scipy.stats as stats
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
                        default= '/Users/romapatel/github/llld-sketch/data/temp/animals/natural/',
            help='')
    parser.add_argument('--eval', '-eval', type=str, default='True',
            help='')
    parser.add_argument('--test_datadir', '-test_datadir', type=str, \
                        default= '/Users/romapatel/github/llld-sketch/data/temp/animals/natural/',
            help='')
    parser.add_argument('--fin_run', '-fin_run', type=str, default='only-reconstr',
            help='')
    return parser.parse_args(args)



# take photo + text, predict photo + text
def model_1(args):
    if os.path.isdir(os.getcwd() + '/results/images/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/images/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/history/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/history/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/files/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/files/' + args.run_name)

    datapath = args.datadir
    #args.batch_size = 2
    args.img_size = 224
    dataset, data_loader = utils.get_dataset(datapath, args.img_size, \
            args.batch_size)
    classes, class_to_idx, idx_to_class = utils.get_classes(dataset)
    word_dim = 300
    label_criterion = nn.CrossEntropyLoss()
    reconstr_criterion = nn.L1Loss()
    #reconstr_criterion = nn.MSELoss()

    model = BimodalDAEImage(300, 2048, n_classes=len(classes))
    cnn = resnet101(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))


    #### change temp
    #word_vecs = utils.get_wvecs_json(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)
    word_vecs = utils.get_word_vectors(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)


    loss_hist, metric_hist = {}, {}
    softmax = nn.Softmax(dim=1)
    for epoch in range(args.epochs):
        print('Epoch %r' %epoch)
        log.info('Epoch %r' %epoch)
        loss_hist[epoch], metric_hist[epoch] = {}, {}
        for batch_idx, (img, target_tensor) in enumerate(data_loader):
            batch_acc, batch_loss = [], {'reconstr': [], 'classification': []}
            target_idxs = target_tensor.data.numpy().tolist()
            target_names = [idx_to_class[idx] for idx in target_idxs]
            target_labels = torch.tensor([[1 if i == idx else 0 for i in \
                    range(len(classes))] for idx in target_idxs], \
                    dtype=torch.long)

            # previously target dist reps
            target_textual = torch.tensor([word_vecs[name] for name in target_names], \
                                            dtype=torch.float32)
            #print('Text', target_textual.size())

            #img_rep = img[0].reshape(1, 3, args.img_size, args.img_size)
            #print(img_rep.size())
            #rep = vgg.forward(img_rep)
            #print(rep.size())
            target_visual = torch.tensor(
                [cnn.forward(
                    img[idx].reshape(1, 3, args.img_size, args.img_size)).data.numpy() for idx in range(len(target_idxs))], dtype=torch.float32
            )

            #print('Visual', target_visual.size())

            n_samples = len(target_idxs)
            optimizer.zero_grad()

            img_reconstr, text_reconstr, hidden = model.forward(target_visual, \
                                                              target_textual)
            textual_loss = reconstr_criterion(text_reconstr, target_textual)
            textual_loss.backward(retain_graph=True)
            visual_loss = reconstr_criterion(img_reconstr, target_visual)
            visual_loss.backward(retain_graph=True)

            #print('Textual reconstr', text_reconstr.size())

            #print('Visual reconstr', img_reconstr.size())
            #print('Hidden', hidden.size())
            preds = softmax(hidden)

            pred_loss = label_criterion(preds, target_tensor)
            pred_loss.backward()

            optimizer.step()

            if epoch%10 == 0:
                state = {'epoch': epoch + 1, 'state_dict': \
                        model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, os.getcwd() + "/model_states/" + args.run_name)


    return

import re
# take photo + text, predict sketch + text
def model_2(args):
    if os.path.isdir(os.getcwd() + '/results/images/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/images/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/history/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/history/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/files/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/files/' + args.run_name)

    datapath = args.datadir
    #########



    #change sketch path, batch size
    sketch_datapath = re.sub('natural', 'sketch', datapath)
    #sketch_datapath = re.sub
    #args.batch_size = 2
    args.img_size = 224
    photo_path = '/data/nlp/sketchy_splits/photo/train/'

    dataset, data_loader = utils.get_photo_sketch_dataset(photo_path, args.img_size, \
            args.batch_size, sketch_datapath)


    classes, class_to_idx, idx_to_class = utils.get_classes(dataset)
    word_dim = 300
    label_criterion = nn.CrossEntropyLoss()
    reconstr_criterion = nn.L1Loss()
    #reconstr_criterion = nn.MSELoss()

    model = BimodalDAEImage(300, 2048, n_classes=len(classes))
    cnn = resnet101(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))
    print('\nDataloader: %r' % (len(data_loader)))


    #### change temp
    #word_vecs = utils.get_wvecs_json(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)
    word_vecs = utils.get_word_vectors(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)


    loss_hist, metric_hist = {}, {}
    softmax = nn.Softmax(dim=1)
    for epoch in range(args.epochs):
        print('Epoch %r' %epoch)
        log.info('Epoch %r' %epoch)
        loss_hist[epoch], metric_hist[epoch] = {}, {}
        #for batch_idx, (img, target_tensor) in enumerate(data_loader):
        for batch_idx, (img, target_tensor, sketch_img) in data_loader:

            batch_acc, batch_loss = [], {'reconstr': [], 'classification': []}
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
                    sketch_img[idx].reshape(1, 3, args.img_size, args.img_size)).data.numpy() for idx in range(len(target_idxs))], dtype=torch.float32
            )

            #print('Visual', target_visual.size())

            n_samples = len(target_idxs)
            optimizer.zero_grad()

            img_reconstr, text_reconstr, hidden = model.forward(target_visual, \
                                                              target_textual)
            textual_loss = reconstr_criterion(text_reconstr, target_textual)
            textual_loss.backward(retain_graph=True)
            visual_loss = reconstr_criterion(img_reconstr, target_visual)
            visual_loss.backward(retain_graph=True)

            #print('Textual reconstr', text_reconstr.size())

            #print('Visual reconstr', img_reconstr.size())
            #print('Hidden', hidden.size())
            preds = softmax(hidden)

            pred_loss = label_criterion(preds, target_tensor)
            pred_loss.backward()

            optimizer.step()

            print(textual_loss); print(visual_loss); print(pred_loss); print()
            if epoch%10 == 0:
                state = {'epoch': epoch + 1, 'state_dict': \
                        model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, os.getcwd() + "/model_states/" + args.run_name)


    return

# take photo + text, predict attr + text
def model_3(args):
    if os.path.isdir(os.getcwd() + '/results/images/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/images/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/history/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/history/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/files/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/files/' + args.run_name)

    datapath = args.datadir
    #args.batch_size = 2
    args.img_size = 224
    photo_path = '/data/nlp/sketchy_splits/photo/train/'

    dataset, data_loader = utils.get_dataset(datapath, args.img_size, \
            args.batch_size)
    classes, class_to_idx, idx_to_class = utils.get_classes(dataset)
    word_dim = 300
    attr_dict, n_attrs = get_attrs(class_to_idx)

    label_criterion = nn.CrossEntropyLoss()
    reconstr_criterion = nn.L1Loss()
    #reconstr_criterion = nn.MSELoss()

    model = BimodalDAEAttr(300, 2048, n_attrs, n_classes=len(classes))
    cnn = resnet101(pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))


    #### change temp
    #word_vecs = utils.get_wvecs_json(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)
    word_vecs = utils.get_word_vectors(os.getcwd() + '/data/files/wvecs.json', classes, word_dim)

    loss_hist, metric_hist = {}, {}
    softmax = nn.Softmax(dim=1)
    for epoch in range(args.epochs):
        print('Epoch %r' %epoch)
        log.info('Epoch %r' %epoch)
        loss_hist[epoch], metric_hist[epoch] = {}, {}
        for batch_idx, (img, target_tensor) in enumerate(data_loader):
            batch_acc, batch_loss = [], {'reconstr': [], 'classification': []}
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
            #print('Text', target_textual.size())

            #img_rep = img[0].reshape(1, 3, args.img_size, args.img_size)
            #print(img_rep.size())
            #rep = vgg.forward(img_rep)
            #print(rep.size())
            target_visual = torch.tensor(
                [cnn.forward(
                    img[idx].reshape(1, 3, args.img_size, args.img_size)).data.numpy() for idx in range(len(target_idxs))], dtype=torch.float32
            )

            #print('Visual', target_visual.size())

            n_samples = len(target_idxs)
            optimizer.zero_grad()

            img_reconstr, text_reconstr, hidden = model.forward(target_visual, \
                                                              target_textual)
            textual_loss = reconstr_criterion(text_reconstr, target_textual)
            textual_loss.backward(retain_graph=True)
            visual_loss = reconstr_criterion(img_reconstr, target_attrs)
            visual_loss.backward(retain_graph=True)

            #print('Textual reconstr', text_reconstr.size())

            #print('Visual reconstr', img_reconstr.size())
            #print('Hidden', hidden.size())
            preds = softmax(hidden)

            pred_loss = label_criterion(preds, target_tensor)
            pred_loss.backward()
            #print(pred_loss)

            optimizer.step()

            if epoch%10 == 0:
                state = {'epoch': epoch + 1, 'state_dict': \
                        model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, os.getcwd() + "/model_states/" + args.run_name)


    return

# unimodal, reconstr photo
def model_4(args):
    return

# unimodal, reconstr sketch
def model_5(args):
    return
if __name__ == '__main__':

    args = sys.argv[1:]
    handled_args = handle_args(args)
    if os.path.isdir(os.getcwd() + '/logs/') is False:
        os.mkdir(os.getcwd() + '/logs/')
    fname = os.getcwd() + '/logs/' + handled_args.run_name + '.log'
    log.basicConfig(filename = fname, format='%(asctime)s: %(message)s', \
                    datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

    #handled_args.fin_run = 'sketch-reconstr'


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




    
