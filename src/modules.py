import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, h_first, h_second, h_third):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(h_first, h_second),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.2),
            nn.Linear(h_second, h_third),
            nn.ReLU(inplace=True),
            )

    def forward(self, input):
        hidden = self.encoder(input)
        return hidden

class Decoder(nn.Module):
    def __init__(self, h_first, h_second, h_third):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(h_first, h_second),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.2),
            nn.Linear(h_second, h_third),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        reconstr = self.decoder(input)
        return reconstr


class BimodalDAEImage(nn.Module):
    def __init__(self, text_dim, img_dim, n_classes):
        super(BimodalDAEImage, self).__init__()
        #self.img_encoder = ImageEncoder(input_dim=img_dim)
        self.img_encoder = Encoder(img_dim, 1024, 100)
        #self.text_encoder = TextEncoder(input_dim=text_dim)
        self.text_encoder = Encoder(text_dim, 200, 100)
        self.img_decoder = Decoder(100, 1024, img_dim)
        #self.text_encoder = TextEncoder(input_dim=text_dim)
        self.text_decoder = Decoder(100, 200, text_dim)
        self.encode_latent = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(200, n_classes),
            nn.ReLU(inplace=True),
        )
        self.decode_latent = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(n_classes, 200),
            nn.ReLU(inplace=True),
        )

        #print(self.img_encoder)

    # take in image, reconstruct image
    def forward(self, img_input, text_input):
        # encode visual component i.e., image features
        img = self.img_encoder.forward(img_input)
        # encode textual i.e., glove
        text = self.text_encoder.forward(text_input)
        # concatenate both and encode to latent (n_classes)
        hidden = self.encode_latent(torch.cat((img, text), dim=1))
        # decode from latent and split into img, text
        reconstr_hidden = self.decode_latent(hidden)
        img_reconstr, text_reconstr = torch.split(reconstr_hidden, 100, dim=1)
        # decode visual component back to image features
        img_reconstr = self.img_decoder.forward(img_reconstr)
        # decode textual component back to text features
        text_reconstr = self.text_decoder.forward(text_reconstr)
        return img_reconstr, text_reconstr, hidden





class BimodalDAEAttr(nn.Module):
    def __init__(self, text_dim, img_dim, attr_dim, n_classes):
        super(BimodalDAEAttr, self).__init__()
        #self.img_encoder = ImageEncoder(input_dim=img_dim)
        self.img_encoder = Encoder(img_dim, 1024, 100)
        #self.text_encoder = TextEncoder(input_dim=text_dim)
        self.text_encoder = Encoder(text_dim, 200, 100)
        self.img_decoder = Decoder(100, 200, attr_dim)
        #self.text_encoder = TextEncoder(input_dim=text_dim)
        self.text_decoder = Decoder(100, 200, text_dim)
        self.encode_latent = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(200, n_classes),
            nn.ReLU(inplace=True),
        )
        self.decode_latent = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(n_classes, 200),
            nn.ReLU(inplace=True),
        )

    # take in image, construct attr
    def forward(self, img_input, text_input):
        # encode visual component i.e., image features
        img = self.img_encoder.forward(img_input)
        # encode textual i.e., glove
        text = self.text_encoder.forward(text_input)
        # concatenate both and encode to latent (n_classes)
        hidden = self.encode_latent(torch.cat((img, text), dim=1))
        # decode from latent and split into img, text
        reconstr_hidden = self.decode_latent(hidden)
        img_reconstr, text_reconstr = torch.split(reconstr_hidden, 100, dim=1)
        # decode visual component back to image features
        img_reconstr = self.img_decoder.forward(img_reconstr)
        # decode textual component back to text features
        text_reconstr = self.text_decoder.forward(text_reconstr)
        return img_reconstr, text_reconstr, hidden

if __name__ == '__main__':
    dae = BimodalDAEImage(300, 4096, n_classes=69)
    img = torch.tensor(np.random.rand(1, 4096), dtype=torch.float32)
    text = torch.tensor(np.random.rand(1, 300), dtype=torch.float32)
    img, text, hidden = dae.forward(img, text)
    print(img.size())
    print(text.size())
    print(hidden.size())

    dae_attr = BimodalDAEAttr(300, 4096, 299, n_classes=69)
    img = torch.tensor(np.random.rand(1, 4096), dtype=torch.float32)
    text = torch.tensor(np.random.rand(1, 300), dtype=torch.float32)
    img, text, hidden = dae_attr.forward(img, text)
    print(img.size())
    print(text.size())
    print(hidden.size())
