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
                        default= '/Users/romapatel/github/sketch-attr/data/temp/animals/natural/',
            help='')
    parser.add_argument('--eval', '-eval', type=str, default='True',
            help='')
    parser.add_argument('--test_datadir', '-test_datadir', type=str, \
                        default= '/Users/romapatel/github/sketch-attr/data/temp/animals/natural/',
            help='')


    return parser.parse_args(args)

def model(args):
    args = handle_args(args)
    if os.path.isdir(os.getcwd() + '/results/images/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/images/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/history/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/history/' + args.run_name)

    if os.path.isdir(os.getcwd() + '/results/files/' + args.run_name) is False:
        os.mkdir(os.getcwd() + '/results/files/' + args.run_name)

    datapath = args.datadir
    dataset, data_loader = utils.get_dataset(datapath, args.img_size, \
            args.batch_size)
    classes, class_to_idx, idx_to_class = utils.get_classes(dataset)

    # todo! change classification to word vectors!
    # todo! change the number of classes to intersect with quickdraws classes!
    #label_criterion = nn.L1Loss()
    label_criterion = nn.CrossEntropyLoss()
    #label_criterion = nn.MultiLabelMarginLoss()
    reconstr_criterion = nn.MSELoss()

    model = DistributedWordLabeller(width=args.img_size, height=args.img_size, label_dim=len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))

    loss_hist, metric_hist = {}, {}
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

            n_samples = len(target_idxs)
            optimizer.zero_grad()

            reconstr, label_dist = model.forward(img)
            labels = model.pred_labels(label_dist)
            #print('Predicted labels'); print(labels)
            #print('True labels'); print(target_tensor)

            reconstr_loss = reconstr_criterion(reconstr, img)
            reconstr_loss.backward(retain_graph=True)
            #print(img)
            #print(label_dist); print(target_tensor)
            
            label_loss = label_criterion(label_dist, target_tensor)
            print(label_loss)
            #continue
            label_loss.backward()
            optimizer.step()
            train_matrix = update_label_matrix(np.zeros((len(classes), len(classes))), \
                                               labels, target_idxs)
            avg_acc, metric_dict = matrix_to_metrics(train_matrix, idx_to_class)
            batch_acc.append(avg_acc)
            batch_loss['reconstr'].append(reconstr_loss.data.numpy().tolist())
            batch_loss['classification'].append(label_loss.data.numpy().tolist())
            loss_hist[epoch][batch_idx] = batch_loss
            metric_hist[epoch][batch_idx] = metric_dict
            log.info('Reconstr loss, classification loss: (%r, %r)' \
                     %(reconstr_loss.data.numpy().tolist(), \
                       label_loss.data.numpy().tolist()))
            log.info('Batch accuracy: (%r)' %(avg_acc))
            if epoch%10 == 0:
                state = {'epoch': epoch + 1, 'state_dict': \
                        model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, os.getcwd() + "/model_states/" + args.run_name)
                save_image(reconstr, os.getcwd() + '/results/images/' \
                           + args.run_name + '/epoch_' + str(epoch) + '.png')

                # save weights
                identity = torch.eye(len(classes))
                # save loss and metrics
                f = open(os.getcwd() + '/results/history/' + args.run_name + \
                         '/loss_hist.json', 'w+')
                f.write(json.dumps(loss_hist))
                f = open(os.getcwd() + '/results/history/' + args.run_name + \
                         '/metric_hist.json', 'w+')
                f.write(json.dumps(metric_hist))


    # todo! don't return from here?
    #return

    matrix = np.zeros((len(classes), len(classes)))
    if args.eval == 'True':
        datapath = args.test_datadir
        dataset, data_loader = utils.get_dataset(datapath, args.img_size, \
            args.batch_size)
        checkpoint = torch.load(os.getcwd() + '/model_states/' + args.run_name)
        model.load_state_dict(checkpoint['state_dict'])

        with torch.no_grad():
            for batch_idx, (img, target_tensor) in enumerate(data_loader):
                target_idxs = target_tensor.data.numpy().tolist()
                target_names = [idx_to_class[idx] for idx in target_idxs]
                target_labels = torch.tensor([[1 if i == idx else 0 for i in \
                    range(len(classes))] for idx in target_idxs], \
                    dtype=torch.float32)

                reconstr, label_dist = model.forward(img)
                labels = model.pred_labels(label_dist)
                matrix = update_label_matrix(matrix, labels, target_idxs)
        avg_acc, metric_dict = matrix_to_metrics(train_matrix, idx_to_class)
        f = open(os.getcwd() + '/results/files/' + args.run_name + \
                         '/matrix.json', 'w+')
        temp = {'matrix': matrix.tolist(), 'metrics': metric_dict, 'avg_acc': avg_acc}
        f.write(json.dumps(temp))


if __name__ == '__main__':

    #run(sketch_idx, img_type, path)
    args = sys.argv[1:]
    handled_args = handle_args(args)
    if os.path.isdir(os.getcwd() + '/logs/') is False:
        os.mkdir(os.getcwd() + '/logs/')
    fname = os.getcwd() + '/logs/' + handled_args.run_name + '.log'
    log.basicConfig(filename = fname, format='%(asctime)s: %(message)s', \
                    datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
    model(args)
    #eval(args)
    


    



    
