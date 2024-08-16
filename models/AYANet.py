import torch
import torch.nn as nn
from models.util import upsample
from models.AYANet_element import *

class AYANet(nn.Module):

    def __init__(self, encoder_arch, decoder_arch):
        super(AYANet, self).__init__()

        self.decoder_arch = decoder_arch
        if encoder_arch == 'double':
            gabor_encoder_arch = 'gaborencoder'
            encoder_arch = 'efficientnet_ayn'
        elif encoder_arch == 'gaborencoder':
            gabor_encoder_arch = 'gaborencoder'

        self.gabor_encoder1, self.gabor_channels = get_encoder(gabor_encoder_arch, pretrained=False)
        self.gabor_encoder2, _ = get_encoder(gabor_encoder_arch, pretrained=False)
    
        self.encoder1, channels = get_encoder(encoder_arch,pretrained=False)
        self.encoder2, _ = get_encoder(encoder_arch,pretrained=False)

        # For double encoder
        channels = [c + g for c, g in zip(channels, self.gabor_channels)]

        # For Gabor encoder only 
        # channels = self.gabor_channels

        _input_addition = 0
        
        # FCM
        self.mtf_module = get_mtf_module(mode='iades', kernel_size=3, channels = channels, input_addition = _input_addition)
        self.decoder = get_decoder(decoder_arch, channels=channels)
        self.classifier = get_classifier(channels[0], 2)

    def forward(self, img):

        img_t0,img_t1 = torch.split(img,3,1)

        # Grayscale image
        rgb = torch.Tensor([0.2989, 0.5870, 0.1140])
        rgb = rgb.view(1,-1,1,1)
        device = img_t0.get_device()
        gray_img_t0 = torch.sum(torch.mul(img_t0, rgb.to(device)), dim=1, keepdim=True).to(device)
        gray_img_t1 = torch.sum(torch.mul(img_t1, rgb.to(device)), dim=1, keepdim=True).to(device)
        
        # Expand grayscale image
        B, C, H, W = gray_img_t0.size()
        Cn = int(self.gabor_channels[0]/4)
        gray_img_t0 = gray_img_t0.expand((B, Cn, H, W))
        gray_img_t1 = gray_img_t1.expand((B, Cn, H, W))

        # Gabor encoder
        gabor_features_t0 = self.gabor_encoder1(gray_img_t0)
        gabor_features_t1 = self.gabor_encoder2(gray_img_t1)

        # CNN encoder
        features_t0 = self.encoder1(img_t0)
        features_t1 = self.encoder2(img_t1)

        # Concatenate features
        combined_features_t0 = []
        combined_features_t1 = []
        for g, f in zip(gabor_features_t0, features_t0):
            f_t0 = torch.cat((g, f), dim=1)
            combined_features_t0.append(f_t0)

        for g, f in zip(gabor_features_t1, features_t1):
            f_t1 = torch.cat((g, f), dim=1)
            combined_features_t1.append(f_t1)
        
        
        features = combined_features_t0 + combined_features_t1
        # Gabor encoder only
        # features = gabor_features_t0 + gabor_features_t1
        

        # FCM
        features_map = self.mtf_module(features)

        pred_ = self.decoder(features_map)
        pred = self.classifier(pred_)

        return pred


