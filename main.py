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

def model_encoder(args):
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
    word_dim = 300
    label_dim = len(classes)
    # todo! change the number of classes to intersect with quickdraws classes!
    #label_criterion = nn.L1Loss()
    label_criterion = nn.CrossEntropyLoss()
    #label_criterion = nn.MultiLabelMarginLoss()
    reconstr_criterion = nn.MSELoss()

    model = DistributedWordLabellerOnly(width=args.img_size, height=args.img_size, \
                                    word_dim=word_dim, label_dim=label_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))

    word_vecs = utils.get_word_vectors('/data/nlp/glove/glove_300d.json', classes, word_dim)

    loss_hist, metric_hist = {}, {}
    for epoch in range(args.epochs):
        print('Epoch %r' %epoch)
        log.info('Epoch %r' %epoch)
        loss_hist[epoch], metric_hist[epoch] = {}, {}
        for batch_idx, (img, target_tensor) in enumerate(data_loader):
            batch_acc, batch_loss = [], {'reconstr': [], 'classification': []}
            target_idxs = target_tensor.data.numpy().tolist()
            target_names = [idx_to_class[idx] for idx in target_idxs]

            target_dist_reps = torch.tensor([word_vecs[name] for name in target_names], \
                                            dtype=torch.float32)
            target_labels = torch.tensor([[1 if i == idx else 0 for i in \
                    range(len(classes))] for idx in target_idxs], \
                    dtype=torch.long)

            n_samples = len(target_idxs)
            optimizer.zero_grad()

            word_dist, label_dist = model.forward(img)
            labels = model.pred_labels(label_dist)

            word_loss=  reconstr_criterion(word_dist, target_dist_reps)
            word_loss.backward(retain_graph=True)

            label_loss = label_criterion(label_dist, target_tensor)
            label_loss.backward()
            optimizer.step()

            train_matrix = update_label_matrix(np.zeros((len(classes), len(classes))), \
                                               labels, target_idxs)
            avg_acc, metric_dict = matrix_to_metrics(train_matrix, idx_to_class)
            batch_acc.append(avg_acc)
            batch_loss['classification'].append(label_loss.data.numpy().tolist())
            loss_hist[epoch][batch_idx] = batch_loss
            metric_hist[epoch][batch_idx] = metric_dict
            log.info('Classification loss: (%r)' \
                     %(label_loss.data.numpy().tolist()))
            log.info('Batch accuracy: (%r)' %(avg_acc))
            if epoch%10 == 0:
                state = {'epoch': epoch + 1, 'state_dict': \
                        model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, os.getcwd() + "/model_states/" + args.run_name)

                # save weights
                identity = torch.eye(len(classes))
                # save loss and metrics
                f = open(os.getcwd() + '/results/history/' + args.run_name + \
                         '/loss_hist.json', 'w+')
                f.write(json.dumps(loss_hist))
                f = open(os.getcwd() + '/results/history/' + args.run_name + \
                         '/metric_hist.json', 'w+')
                f.write(json.dumps(metric_hist))



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

                word_dist, label_dist = model.forward(img)
                labels = model.pred_labels(label_dist)
                matrix = update_label_matrix(matrix, labels, target_idxs)
        avg_acc, metric_dict = matrix_to_metrics(train_matrix, idx_to_class)
        f = open(os.getcwd() + '/results/files/' + args.run_name + \
                         '/matrix.json', 'w+')
        temp = {'matrix': matrix.tolist(), 'metrics': metric_dict, 'avg_acc': avg_acc}
        f.write(json.dumps(temp))


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
    word_dim = 300
    label_dim = len(classes)
    # todo! change the number of classes to intersect with quickdraws classes!
    #label_criterion = nn.L1Loss()
    label_criterion = nn.CrossEntropyLoss()
    #label_criterion = nn.MultiLabelMarginLoss()
    reconstr_criterion = nn.MSELoss()

    model = DistributedWordLabeller(width=args.img_size, height=args.img_size, \
                                    word_dim=word_dim, label_dim=label_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))

    #word_vecs = {i: [0, 0, 0] for i in range(300)}
    word_vecs = utils.get_word_vectors('/data/nlp/glove/glove_300d.json', classes, word_dim)

    loss_hist, metric_hist = {}, {}
    for epoch in range(args.epochs):
        print('Epoch %r' %epoch)
        log.info('Epoch %r' %epoch)
        loss_hist[epoch], metric_hist[epoch] = {}, {}
        for batch_idx, (img, target_tensor) in enumerate(data_loader):
            batch_acc, batch_loss = [], {'reconstr': [], 'classification': []}
            target_idxs = target_tensor.data.numpy().tolist()
            target_names = [idx_to_class[idx] for idx in target_idxs]

            target_dist_reps = torch.tensor([word_vecs[name] for name in target_names], \
                                            dtype=torch.float32)
            target_labels = torch.tensor([[1 if i == idx else 0 for i in \
                    range(len(classes))] for idx in target_idxs], \
                    dtype=torch.long)

            n_samples = len(target_idxs)
            optimizer.zero_grad()

            reconstr, word_dist, label_dist = model.forward(img)
            labels = model.pred_labels(label_dist)

            reconstr_loss = reconstr_criterion(reconstr, img)
            reconstr_loss.backward(retain_graph=True)

            word_loss=  reconstr_criterion(word_dist, target_dist_reps)
            word_loss.backward(retain_graph=True)

            label_loss = label_criterion(label_dist, target_tensor)
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

                reconstr, word_dist, label_dist = model.forward(img)
                labels = model.pred_labels(label_dist)
                matrix = update_label_matrix(matrix, labels, target_idxs)
        avg_acc, metric_dict = matrix_to_metrics(train_matrix, idx_to_class)
        f = open(os.getcwd() + '/results/files/' + args.run_name + \
                         '/matrix.json', 'w+')
        temp = {'matrix': matrix.tolist(), 'metrics': metric_dict, 'avg_acc': avg_acc}
        f.write(json.dumps(temp))

def eval(args):
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
    word_dim = 300
    label_dim = len(classes)
    # todo! change the number of classes to intersect with quickdraws classes!
    #label_criterion = nn.L1Loss()
    label_criterion = nn.CrossEntropyLoss()
    #label_criterion = nn.MultiLabelMarginLoss()
    reconstr_criterion = nn.MSELoss()

    model = DistributedWordLabeller(width=args.img_size, height=args.img_size, \
                                    word_dim=word_dim, label_dim=label_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)
    print('\nNum classes: %r, num images: %r' % (len(classes), len(dataset)))

    #word_vecs = {i: [0, 0, 0] for i in range(300)}
    word_vecs = utils.get_word_vectors('/data/nlp/glove/glove_300d.json', classes, word_dim)

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
                print(target_names)
                target_labels = torch.tensor([[1 if i == idx else 0 for i in \
                    range(len(classes))] for idx in target_idxs], \
                    dtype=torch.float32)

                reconstr, word_dist, label_dist = model.forward(img)
                labels = model.pred_labels(label_dist)
                matrix = update_label_matrix(matrix, labels, target_idxs)
        avg_acc, metric_dict = matrix_to_metrics(train_matrix, idx_to_class)

        print(avg_acc); print(metric_dict)
        return
        f = open(os.getcwd() + '/results/files/' + args.run_name + \
                         '/matrix.json', 'w+')
        temp = {'matrix': matrix.tolist(), 'metrics': metric_dict, 'avg_acc': avg_acc}
        f.write(json.dumps(temp))


# get matrix weights

import operator
def extract_weights(args):
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
    classes, class_to_idx, idx_to_class = utils.get_temp_classes(dataset)
    word_dim = 300
    label_dim = len(classes)


    print('Model: ', args.run_name)
    model = DistributedWordLabeller(width=args.img_size, height=args.img_size, \
                                    word_dim=word_dim, label_dim=label_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=1e-5)

    # get the matrix and classify the matrix

    f = open(os.getcwd() + '/results/files/' + args.run_name + '/matrix.json', 'r')
    for line in f: temp = json.loads(line)

    matrix = temp['matrix']

    #print(len(matrix))
    #print(matrix)
    #print(matrix[0][0])
    print(idx_to_class)
    avg_acc, metric_dict = matrix_to_metrics(matrix, idx_to_class)
    #print(metric_dict)
    #print(avg_acc)


    print(idx_to_class)
    for idx in metric_dict:
        print(idx_to_class[idx])
        print(metric_dict[idx])

    return


    # get the weights from the linear layer, this forms the matrix, classify this matrix,
    matrix = np.zeros((len(classes), len(classes)))
    identity = torch.eye(300)

    m = model.label_classifier(identity)
    m = torch.transpose(m, 0, 1)    #print(m)
    f = open(os.getcwd() + '/results/files/sketch_sketchy/' + 'idx_to_class.json', 'r')
    for line in f: idx_to_class = json.loads(line)

    rep_dict = {}
    for i in range(len(m)):
        class_name = idx_to_class[str(i)]
        rep_dict[class_name] = m[i].data.numpy()

    f = open(os.getcwd() + '/data/files/wvecs.json', 'r')
    for line in f: wvecs = json.loads(line)

    for name in wvecs:
        wvecs[name] = np.array(wvecs[name])
    classes = list(rep_dict.keys())
    sims = []

    def get_sim(class_1, class_2, rep_dict):
        dot_product = rep_dict[class_1].reshape(-1, 1) * rep_dict[class_2].reshape(-1, 1)
        #print(dot_product)
        norm_1 = (np.linalg.norm(rep_dict[class_1].reshape(-1, 1)))
        norm_2 = (np.linalg.norm(rep_dict[class_2].reshape(-1, 1)))
        cos_sim = np.sum(dot_product)/(norm_1*norm_2)
        return cos_sim

    for (class_1, class_2) in itertools.product(classes, classes):
        pair = class_1 + '#' + class_2
        cos_sim = get_sim(class_1, class_2, rep_dict)
        word_sim = get_sim(class_1, class_2, wvecs)

        #print(cos_sim)
        sims.append((pair, round(cos_sim, 3), round(word_sim, 3)))

    sorted_sims = sorted(sims, key=operator.itemgetter(1), reverse=True)

    for item in sorted_sims:
        if item[1] > 0.9: continue
        print(item)

    print(len(sorted_sims))


    def get_corrs(pair_tuples):
        temp = 'sailboat,piano,sheep,pistol,snail,harp,cat,rocket,cannon,rabbit'
        temp = temp.split(',')
        f = open(os.getcwd() + '/data/files/sem-vis-sketchy.tsv', 'r')
        lines = [line.strip().split('\t') for line in f.readlines()]

        pairs = {item[0]:item[1] for item in pair_tuples}
        vals = []
        for line in lines:
            class_1, class_2, sem, vis = line[0], line[1], float(line[2]), float(line[3])
            if class_1 in temp or class_2 in temp: continue
            pair = class_1 + '#' + class_2
            if pair in pairs:
                vals.append((pair, sem, vis, pairs[pair]))


        for val in vals:
            print(val)

        cos_list = [item[-1] for item in vals]
        sem_list = [item[1] for item in vals]
        vis_list = [item[2] for item in vals]

        #print(vals)
        spearman_sem = stats.spearmanr(cos_list, sem_list)
        spearman_vis = stats.spearmanr(cos_list, vis_list)

        pearson_sem = stats.pearsonr(cos_list, sem_list)
        pearson_vis = stats.pearsonr(cos_list, vis_list)

        print('Semantic: Pearson: %f, Spearman: %f' %(round(pearson_sem[0], 3), \
                                                      round(spearman_sem[0], 3)))
        print('Visual: Pearson: %f, Spearman: %f' %(round(pearson_vis[0], 3), \
                                                      round(spearman_vis[0], 3)))

    print('\n\nImage!')
    get_corrs([(item[0], item[1]) for item in sims])
    #print('Word!')

    #get_corrs([(item[0], item[2]) for item in sims])





    return

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
                print(target_names)
                target_labels = torch.tensor([[1 if i == idx else 0 for i in \
                    range(len(classes))] for idx in target_idxs], \
                    dtype=torch.float32)

                reconstr, word_dist, label_dist = model.forward(img)
                labels = model.pred_labels(label_dist)
                matrix = update_label_matrix(matrix, labels, target_idxs)
        avg_acc, metric_dict = matrix_to_metrics(matrix, idx_to_class)

    print(avg_acc); print(metric_dict)
    return


if __name__ == '__main__':

    #run(sketch_idx, img_type, path)
    args = sys.argv[1:]
    handled_args = handle_args(args)
    if os.path.isdir(os.getcwd() + '/logs/') is False:
        os.mkdir(os.getcwd() + '/logs/')
    fname = os.getcwd() + '/logs/' + handled_args.run_name + '.log'
    log.basicConfig(filename = fname, format='%(asctime)s: %(message)s', \
                    datefmt='%m/%d %I:%M:%S %p', level=log.INFO)
    #model(args)
    extract_weights(args)
    #model_encoder(args)
    


    



    
