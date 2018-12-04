import torch
import torch.nn as nn
import torch.nn.functional as F
#from PIL import Image

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            #nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

    def forward(self, input):
        hidden = self.encoder(input)
        return hidden

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            #nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, input):
        reconstr = self.decoder(input)
        return reconstr


class TextEncoder:
    def __init__(self, input_id, hidden_dim=1000):
        self.input_id = input_id
        
    def forward(self):
        rep = torch.tensor([i/1000.0 for i in range(1000)])
        return rep

class Labeller(nn.Module):
    def __init__(self, width, height, label_dim):
        super(Labeller, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8*21*21, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, label_dim),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def bottleneck(self, rep):
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    def forward(self, input):
        encoded_orig = self.encoder.forward(input)
        reconstr = self.decoder.forward(encoded_orig)

        encoded_reconstr = self.encoder.forward(reconstr)
        labels = self.classifier(encoded_reconstr.view(encoded_reconstr.size(0), -1))

        return reconstr, labels
    
    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)
        return indices[-k:]

    def pred_labels(self, label_dist):
        return torch.argmax(label_dist, dim=1)

# Labeller with attributes, old version
class OldLabeller(nn.Module):
    def __init__(self, width, height, attr_dict, label_dim, attr_dim):
        super(OldLabeller, self).__init__()
        self.encoder = Encoder()
        self.flattened_dim = 3528
        self.attr_transform = nn.Linear(self.flattened_dim, attr_dim)
        self.label_transform = nn.Linear(attr_dim, label_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def bottleneck(self, rep):
        states = self.goal_states()        
        transform = nn.Linear(self.hidden_dim, len(states))
        rep = transform(rep)
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    def forward(self, input):
        hidden = self.encoder.forward(input)
        attr_weights, attr = self.attributes(hidden)
        labels = self.labels(attr_weights)

        return attr, labels
    
    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)

        return indices[-k:]
    
    def attributes(self, rep):
        rep = rep.view(rep.size(0), -1)
        attr_weights = self.attr_transform(rep)
        attr_dist = self.sigmoid(attr_weights)

        return attr_weights, attr_dist

    def labels(self, rep):
        rep = rep.view(rep.size(0), -1)
        label_weights = self.sigmoid(self.label_transform(rep))
        #label_weights = self.label_transform(rep)
        label_dist = self.softmax(label_weights)
        return label_dist

    def pred_labels(self, label_dist):
        return torch.argmax(label_dist, dim=1)

    def pred_attrs(self, attr_dist):
        threshold = 0.7
        #values, indices = attr_dist.sort(dim=0) 
        idxs = []
        for row in attr_dist:
            row_idxs = (row>=threshold).nonzero()
            row_idxs = row_idxs.data.numpy().tolist()
            idxs.append([item[0] for item in row_idxs])

        return idxs


class DistributedWordLabeller(nn.Module):
    def __init__(self, width, height, word_dim, label_dim):
        super(DistributedWordLabeller, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8*21*21, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, word_dim),
        )

        self.label_classifier = nn.Linear(word_dim, label_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def bottleneck(self, rep):
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    def forward(self, input):
        encoded_orig = self.encoder.forward(input)
        reconstr = self.decoder.forward(encoded_orig)

        encoded_reconstr = self.encoder.forward(reconstr)
        words = self.classifier(encoded_reconstr.view(encoded_reconstr.size(0), -1))
        labels = self.label_classifier(words)
        return reconstr, words, labels

    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)
        return indices[-k:]

    def classify_words(self, label_dist):
        return None

    def pred_labels(self, label_dist):
        return torch.argmax(label_dist, dim=1)

class DistributedWordLabellerOnly(nn.Module):
    def __init__(self, width, height, word_dim, label_dim):
        super(DistributedWordLabellerOnly, self).__init__()
        self.encoder = Encoder()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8*21*21, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, word_dim),
        )

        self.label_classifier = nn.Linear(word_dim, label_dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def bottleneck(self, rep):
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    # define the number of things
    def forward(self, input):
        encoded_orig = self.encoder.forward(input)
        #reconstr = self.decoder.forward(encoded_orig)
        #encoded_reconstr = self.encoder.forward(reconstr)
        words = self.classifier(encoded_orig.view(encoded_orig.size(0), -1))
        labels = self.label_classifier(words)
        return words, labels

    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)
        return indices[-k:]

    def classify_words(self, label_dist):
        return None

    def pred_labels(self, label_dist):
        return torch.argmax(label_dist, dim=1)


class DistributedWordOnly(nn.Module):
    def __init__(self, width, height, word_dim, label_dim):
        super(DistributedWordOnly, self).__init__()
        self.encoder = Encoder()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8*21*21, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, word_dim),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def bottleneck(self, rep):
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    # define the number of things
    def forward(self, input):
        encoded_orig = self.encoder.forward(input)
        #reconstr = self.decoder.forward(encoded_orig)
        #encoded_reconstr = self.encoder.forward(reconstr)
        words = self.classifier(encoded_orig.view(encoded_orig.size(0), -1))
        return words, []

    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)
        return indices[-k:]

    def classify_words(self, label_dist):
        return None

    def pred_labels(self, label_dist):
        return torch.argmax(label_dist, dim=1)


class DistributedAttrOnly(nn.Module):
    def __init__(self, width, height, word_dim, label_dim):
        super(DistributedAttrOnly, self).__init__()
        self.encoder = Encoder()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8*21*21, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, word_dim),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def bottleneck(self, rep):
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    # define the number of things
    def forward(self, input):
        encoded_orig = self.encoder.forward(input)
        #reconstr = self.decoder.forward(encoded_orig)
        #encoded_reconstr = self.encoder.forward(reconstr)
        words = self.classifier(encoded_orig.view(encoded_orig.size(0), -1))
        return words, []

    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)
        return indices[-k:]

    def classify_words(self, label_dist):
        return None

    def pred_labels(self, label_dist):
        return torch.argmax(label_dist, dim=1)
    
class DistributedWordOnlyReconstr(nn.Module):
    def __init__(self, width, height, word_dim, label_dim):
        super(DistributedWordOnlyReconstr, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(8*21*21, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, word_dim),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def bottleneck(self, rep):
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    # define the number of things
    def forward(self, input):
        encoded_orig = self.encoder.forward(input)
        reconstr = self.decoder.forward(encoded_orig)
        #encoded_reconstr = self.encoder.forward(reconstr)
        words = self.classifier(encoded_orig.view(encoded_orig.size(0), -1))
        return words, reconstr

    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)
        return indices[-k:]

    def classify_words(self, label_dist):
        return None

    def pred_labels(self, label_dist):
        return torch.argmax(label_dist, dim=1)


class ReconstrOnly(nn.Module):
    def __init__(self, width, height, word_dim, label_dim):
        super(ReconstrOnly, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def bottleneck(self, rep):
        rep = nn.functional.softmax(rep, dim=0)
        return rep

    # define the number of things
    def forward(self, input):
        encoded_orig = self.encoder.forward(input)
        reconstr = self.decoder.forward(encoded_orig)
        #encoded_reconstr = self.encoder.forward(reconstr)
        return [], reconstr

    def get_encoded(self, input):
        encoded_orig = self.encoder.forward(input)
        return encoded_orig

    def ranker(self, rep, k):
        values, indices = rep.sort(dim=0)
        return indices[-k:]

    def classify_words(self, label_dist):
        return None

    def pred_labels(self, label_dist):
        return torch.argmax(label_dist, dim=1)
