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
    
        # self.encoder1, channels = get_encoder(encoder_arch,pretrained=False)
        # self.encoder2, _ = get_encoder(encoder_arch,pretrained=False)

        # For double encoder
        # channels = [c + g for c, g in zip(channels, self.gabor_channels)]
        # print('Channels')
        # print(channels)

        # For Gabor encoder only 
        channels = self.gabor_channels

        # With gabor fusion module
        # _input_addition = 8
        # Without gabor fusion module
        _input_addition = 0
        
        # MTF original
        # self.mtf_module = get_mtf_module(mode='iade', kernel_size=3, channels = channels, input_addition = _input_addition)
        # MTF version 2
        self.mtf_module = get_mtf_module(mode='iades', kernel_size=3, channels = channels, input_addition = _input_addition)
        # Only difference module
        # self.mtf_module = get_mtf_module(mode='ade', kernel_size=3, channels = channels, input_addition = _input_addition)

        self.decoder = get_decoder(decoder_arch, channels=channels)

        self.classifier = get_classifier(channels[0], 2)
        # self.classifier = nn.Conv2d(channels[0], 2, 1, padding=0, stride=1)
        # self.bn = nn.BatchNorm2d(channels[0])
        # self.relu = nn.ReLU(inplace=True)

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

        # gfn_features_t0, gab_features_t0 = self.gfn1(img_t0)
        # gfn_features_t1, gab_features_t1 = self.gfn2(img_t1)

        # Use concat GFN features + original image
        # features_t0 = self.encoder1(gfn_features_t0)
        # features_t1 = self.encoder2(gfn_features_t1)

        # Use only original image
        # features_t0 = self.encoder1(img_t0)
        # features_t1 = self.encoder2(img_t1)

        # Concatenate features
        # combined_features_t0 = []
        # combined_features_t1 = []
        # for g, f in zip(gabor_features_t0, features_t0):
        #     f_t0 = torch.cat((g, f), dim=1)
        #     combined_features_t0.append(f_t0)

        # for g, f in zip(gabor_features_t1, features_t1):
        #     f_t1 = torch.cat((g, f), dim=1)
        #     combined_features_t1.append(f_t1)
        

        # print('Gabor encoder')
        # for f in gabor_features_t0:
        #     print(f.size())
        # print('EfficientNet')
        # for f in combined_features_t0:
        #     print(f.size())
        
        # features = features_t0 + features_t1
        # features = combined_features_t0 + combined_features_t1
        features = gabor_features_t0 + gabor_features_t1
        # Double encoders
        # enc_features = features_t0 + features_t1
        # gab_features = gabor_features_t0 + gabor_features_t1

        # features_map = self.attention_module(features)

        # gab_features = []
        # gab_features.append(gfn_features_t0)
        # gab_features.append(gfn_features_t1)
        # features = self.gabor_fusion(gab_features, features)
        
        # MTFv2
        features_map = self.mtf_module(features)
        # MTFv3
        # features_map = self.mtf_module(enc_features, gab_features)

        pred_ = self.decoder(features_map)
        if self.decoder_arch == 'cf':
            pred = pred_
        else:
            # pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
            # pred_ = self.bn(pred_)
            # pred_ = upsample(pred_,[pred_.size()[2]*2, pred_.size()[3]*2])
            # pred_ = self.relu(pred_)
            pred = self.classifier(pred_)

        # return pred, gabor_features_t0, gabor_features_t1
        return pred


