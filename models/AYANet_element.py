import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from models.util import *
from models.GConv import GDConv
from models.EfficientNet import efficientnet_ayn

__all__ = ['get_encoder', 'get_decoder', 'get_mtf_module', 'get_classifier']

def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)

def gabdconv7x7(in_channels, M, n_scale, stride=1, expand=False):
    return GDConv(in_channels, in_channels, kernel_size=7, M=M, nScale=n_scale, stride=stride,
                    padding=3, dilation=1, groups=in_channels, bias=False, expand=expand, padding_mode='zeros')


def gabor_block_function_factory(conv, gconv, norm, relu=None):
    def block_function(x):
        if conv is not None:
            x = conv(x)
        x = gconv(x)
        if norm is not None:
            x = norm(x)
        if relu is not None:
            x = relu(x)
        return x
    return block_function

def do_efficient_fwd(block_f,x,efficient):
    if efficient and x.requires_grad:
        return cp.checkpoint(block_f,x)
    else:
        return block_f(x)


class GaborEncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, orientation, scale, conv1, expand = False, downsample = None, efficient=True, use_bn=True):
        super(GaborEncoderBlock, self).__init__()

        self.efficient = efficient
        if conv1:
            self.conv1x1 = nn.Conv2d(in_c, out_c*orientation, kernel_size=1, bias=False)
        else:
            self.conv1x1 = None
        if expand:
            self.conv_scale1 = gabdconv7x7(in_c, orientation, scale[0], stride=1, expand=True)
        else:
            self.conv_scale1 = gabdconv7x7(out_c, orientation, scale[0], stride=1)
        self.bn1 = nn.BatchNorm2d(out_c*orientation) if use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.conv_scale2 = gabdconv7x7(out_c, orientation, scale[1], stride=2)
        self.bn2 = nn.BatchNorm2d(out_c*orientation) if use_bn else None


    def forward(self, x):
        block_f1 = gabor_block_function_factory(self.conv1x1, self.conv_scale1, self.bn1, self.relu)
        block_f2 = gabor_block_function_factory(None, self.conv_scale2,self.bn2)

        out = do_efficient_fwd(block_f1,x,self.efficient)
        out = do_efficient_fwd(block_f2,out,self.efficient)
        relu_out = self.relu(out)

        return relu_out, out


class GaborEncoder(nn.Module):
    def __init__(self, input_channels=8, channels=[32,48,80,160]):
        super(GaborEncoder, self).__init__()
        orientation = 4
        scale = [1, 2]
        g_channels = [int(i/orientation) for i in channels]

        self.block1 = GaborEncoderBlock(input_channels, g_channels[0], orientation, scale, conv1=False, expand=True)
        self.block2 = GaborEncoderBlock(channels[0], g_channels[1], orientation, scale, conv1=True)
        self.block3 = GaborEncoderBlock(channels[1], g_channels[2], orientation, scale, conv1=True)
        self.block4 = GaborEncoderBlock(channels[2], g_channels[3], orientation, scale, conv1=True)

    def forward(self, x):
        features = []
        x, skip = self.block1(x)
        features += [skip]
        x, skip = self.block2(x)
        features += [skip]
        x, skip = self.block3(x)
        features += [skip]
        x, skip = self.block4(x)
        features += [skip]
        return features


class Decoder(nn.Module):

    M = 4
    
    def __init__(self,channels=[64,128,256,512]):
        super(Decoder, self).__init__()

        #channels = [i*self.M for i in channels]
        # self.upsample1 = Upsample(num_maps_in=channels[3]*2, skip_maps_in=channels[3], num_maps_out=channels[3])
        # self.upsample2 = Upsample(num_maps_in=channels[2]*2, skip_maps_in=channels[2], num_maps_out=channels[2])
        # self.upsample3 = Upsample(num_maps_in=channels[1]*2, skip_maps_in=channels[1], num_maps_out=channels[1])
        # self.upsample4 = Upsample(num_maps_in=channels[0]*2, skip_maps_in=channels[0], num_maps_out=channels[0])

        self.upsample1 = Upsample(num_maps_in=channels[3]*2, skip_maps_in=channels[3], num_maps_out=channels[3])
        self.upsample2 = Upsample(num_maps_in=channels[3], skip_maps_in=channels[2], num_maps_out=channels[2])
        self.upsample3 = Upsample(num_maps_in=channels[2], skip_maps_in=channels[1], num_maps_out=channels[1])
        self.upsample4 = Upsample(num_maps_in=channels[1], skip_maps_in=channels[0], num_maps_out=channels[0])

    def forward(self, features_map):
        
        x = features_map[0]
        x = self.upsample1(x, features_map[1])
        x = self.upsample2(x, features_map[2])
        x = self.upsample3(x, features_map[3])
        x = self.upsample4(x, features_map[4])
        return x

class Decoder2(nn.Module):

    M = 4

    def __init__(self,channels=[64,128,256,512]):
        super(Decoder2, self).__init__()

        self.upsample1 = Upsample(num_maps_in=channels[3]*2, skip_maps_in=channels[3]*2, num_maps_out=channels[3])
        self.upsample2 = Upsample(num_maps_in=channels[3], skip_maps_in=channels[2]*2, num_maps_out=channels[2])
        self.upsample3 = Upsample(num_maps_in=channels[2], skip_maps_in=channels[1]*2, num_maps_out=channels[1])
        self.upsample4 = Upsample(num_maps_in=channels[1], skip_maps_in=channels[0]*2, num_maps_out=channels[0])

    def forward(self, features_map):

        x = features_map[0]
        x = self.upsample1(x, features_map[1])
        x = self.upsample2(x, features_map[2])
        x = self.upsample3(x, features_map[3])
        x = self.upsample4(x, features_map[4])
        return x


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class MTFModule3(nn.Module):
    def __init__(self, mode='iade', kernel_size=1, channels = [64,128,256,512], input_addition = 0):
        super(MTFModule3, self).__init__()

        self.mtf_layer1 = MTF2(channels[0], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer2 = MTF2(channels[1], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer3 = MTF2(channels[2], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer4 = MTF2(channels[3], mode, kernel_size=3, input_addition = input_addition)

        self.downsample1 = conv3x3(channels[0], channels[1], stride=2)
        self.downsample2 = conv3x3(channels[1]*2, channels[2], stride=2)
        self.downsample3 = conv3x3(channels[2]*2, channels[3], stride=2)

    def forward(self, enc_features, gab_features):
        enc_features_t0, enc_features_t1 = enc_features[:4], enc_features[4:]
        gab_features_t0, gab_features_t1 = gab_features[:4], gab_features[4:]

        e_mtf1 = self.mtf_layer1(enc_features_t0[0], enc_features_t1[0])
        e_mtf2 = self.mtf_layer2(enc_features_t0[1], enc_features_t1[1])
        e_mtf3 = self.mtf_layer3(enc_features_t0[2], enc_features_t1[2])
        e_mtf4 = self.mtf_layer4(enc_features_t0[3], enc_features_t1[3])

        g_mtf1 = self.mtf_layer1(gab_features_t0[0], gab_features_t1[0])
        g_mtf2 = self.mtf_layer2(gab_features_t0[1], gab_features_t1[1])
        g_mtf3 = self.mtf_layer3(gab_features_t0[2], gab_features_t1[2])
        g_mtf4 = self.mtf_layer4(gab_features_t0[3], gab_features_t1[3])

        # concat_mtf1 = torch.cat((e_mtf1, g_mtf1), dim=1)
        # concat_mtf2 = torch.cat((e_mtf2, g_mtf2), dim=1)
        # concat_mtf3 = torch.cat((e_mtf3, g_mtf3), dim=1)
        # concat_mtf4 = torch.cat((e_mtf4, g_mtf4), dim=1)

        downsampled_mtf1 = self.downsample1(e_mtf1)
        cat_mtf2 = torch.cat([downsampled_mtf1,e_mtf2], 1)
        downsampled_mtf2 = self.downsample2(cat_mtf2)
        cat_mtf3 = torch.cat([downsampled_mtf2, e_mtf3], 1)
        downsampled_mtf3 = self.downsample3(cat_mtf3)
        final_mtf_map = torch.cat([downsampled_mtf3, e_mtf4], 1)

        # features_map = [final_mtf_map, concat_mtf4, concat_mtf3, concat_mtf2, concat_mtf1]
        features_map = [final_mtf_map, g_mtf4, g_mtf3, g_mtf2, g_mtf1]
        return features_map

class MTFModule2(nn.Module):
    def __init__(self, mode='iade', kernel_size=1, channels = [64,128,256,512], input_addition = 0):
        super(MTFModule2, self).__init__()

        self.mtf_layer1 = MTF2(channels[0], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer2 = MTF2(channels[1], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer3 = MTF2(channels[2], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer4 = MTF2(channels[3], mode, kernel_size=3, input_addition = input_addition)

        self.downsample1 = conv3x3(channels[0], channels[1], stride=2)
        self.downsample2 = conv3x3(channels[1]*2, channels[2], stride=2)
        self.downsample3 = conv3x3(channels[2]*2, channels[3], stride=2)

        # self.downsample1 = conv3x3(channels[0], channels[1], stride=2)
        # self.downsample2 = conv3x3(channels[2], channels[2], stride=2)
        # self.downsample3 = conv3x3(channels[3], channels[3], stride=2)

    def forward(self, features):
        features_t0, features_t1 = features[:4], features[4:]

        mtf1 = self.mtf_layer1(features_t0[0], features_t1[0])
        mtf2 = self.mtf_layer2(features_t0[1], features_t1[1])
        mtf3 = self.mtf_layer3(features_t0[2], features_t1[2])
        mtf4 = self.mtf_layer4(features_t0[3], features_t1[3])

        downsampled_mtf1 = self.downsample1(mtf1)
        cat_mtf2 = torch.cat([downsampled_mtf1, mtf2], 1)
        downsampled_mtf2 = self.downsample2(cat_mtf2)
        cat_mtf3 = torch.cat([downsampled_mtf2, mtf3], 1)
        downsampled_mtf3 = self.downsample3(cat_mtf3)
        final_mtf_map = torch.cat([downsampled_mtf3, mtf4], 1)

        features_map = [final_mtf_map, mtf4, mtf3, mtf2, mtf1]
        return features_map

class MTF2(nn.Module):
    def __init__(self, channel, mode='iades', kernel_size=1, input_addition = 0):
        super(MTF2, self).__init__()
        assert mode in ['iades', 'iadespr', 'ad', 'ade']
        self.mode = mode
        self.channel = channel
        self.input_addition = input_addition
        dist_channel = 0
        if 'iades' in mode:
            l = 6
        elif mode == 'ad':
            l = 2
        elif mode == 'ade':
            l = 3
        if mode == 'iadespr':
            dist_channel += 3

        # self.mlp = MLP(l*(self.channel+self.input_addition), self.channel)
        self.conv1x1 = nn.Conv2d(l*(self.channel+self.input_addition)+dist_channel, self.channel, kernel_size=1, stride=1, bias=False)
        
        self.conv = conv3x3(self.channel, self.channel)
        self.bn = nn.BatchNorm2d(self.channel)
        self.relu = nn.ReLU(inplace=True)

        # print("MTF: mode: {} kernel_size: {}".format(self.mode, kernel_size))

    def forward(self, f0, f1):
        #t0 = self.conv(f0)
        #t1 = self.conv(f1)

        if 'i' in self.mode:
            info = torch.cat((f0, f1), dim=1)
        if 'a' in self.mode:
            appear = f1 - f0
        if 'd' in self.mode:
            disappear = f0 - f1
        if 'e' in self.mode:
            exchange = torch.max(f0, f1) - torch.min(f0, f1)
        if 's' in self.mode :
            summed = f0 + f1
        if 'p' in self.mode:
            dot_product = torch.sum(f0*f1, dim=1, keepdim=True)
        if 'r' in self.mode:
            f0_var = torch.var(f0, dim=1, keepdim=True)
            f0_ref_var = torch.var(f0-f1, dim=1, keepdim=True)
            snr_f0 = f0_ref_var / f0_var

            f1_var = torch.var(f1, dim=1, keepdim=True)
            f1_ref_var = torch.var(f1-f0, dim=1, keepdim=True)
            snr_f1 = f1_ref_var / f1_var

            snr = torch.cat((snr_f0, snr_f1), dim=1)

        if self.mode == 'iades':
            f = torch.cat((info, appear, disappear, exchange, summed), dim=1)
        elif self.mode == 'ad':
            f = torch.cat((appear, disappear), dim=1)
        elif self.mode == 'ade':
            f = torch.cat((appear, disappear, exchange), dim=1)
        elif self.mode == 'iadespr':
            f = torch.cat((info, appear, disappear, exchange, summed, dot_product, snr), dim=1)


        f = self.conv1x1(f)
        # f = self.bn(f) #v2
        # f = self.relu(f) #v2
        f = self.conv(f)
        f = self.bn(f)
        f = self.relu(f)

        return f

class MTFModule(nn.Module):
    def __init__(self, mode='iade', kernel_size=1, channels = [64,128,256,512], input_addition = 0):
        super(MTFModule, self).__init__()
        
        self.mtf_layer1 = MTF(channels[0], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer2 = MTF(channels[1], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer3 = MTF(channels[2], mode, kernel_size=3, input_addition = input_addition)
        self.mtf_layer4 = MTF(channels[3], mode, kernel_size=3, input_addition = input_addition)

        self.downsample1 = conv3x3(channels[0], channels[1], stride=2)
        self.downsample2 = conv3x3(channels[1]*2, channels[2], stride=2)
        self.downsample3 = conv3x3(channels[2]*2, channels[3], stride=2)
    
    def forward(self, features):
        features_t0, features_t1 = features[:4], features[4:]
        
        mtf1 = self.mtf_layer1(features_t0[0], features_t1[0])
        mtf2 = self.mtf_layer2(features_t0[1], features_t1[1])
        mtf3 = self.mtf_layer3(features_t0[2], features_t1[2])
        mtf4 = self.mtf_layer4(features_t0[3], features_t1[3])
        
        downsampled_mtf1 = self.downsample1(mtf1)
        cat_mtf2 = torch.cat([downsampled_mtf1, mtf2], 1)
        downsampled_mtf2 = self.downsample2(cat_mtf2)
        cat_mtf3 = torch.cat([downsampled_mtf2, mtf3], 1)
        downsampled_mtf3 = self.downsample3(cat_mtf3)
        final_mtf_map = torch.cat([downsampled_mtf3, mtf4], 1)
        
        features_map = [final_mtf_map, mtf4, mtf3, mtf2, mtf1]
        return features_map
    
class MTF(nn.Module):
    def __init__(self, channel, mode='iade', kernel_size=1, input_addition = 0):
        super(MTF, self).__init__()
        assert mode in ['i', 'a', 'd', 'e', 'ia', 'id', 'ie', 'iae', 'ide', 'iad', 'iade', 'iadec', 'i2ade', 'iad2e', 'i2ad2e', 'i2d']
        self.mode = mode
        self.channel = channel
        self.relu = nn.ReLU(inplace=True)
        self.input_addition = input_addition
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1
        if 'i2' in mode:
            self.i0 = nn.Conv2d(self.channel+self.input_addition, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            self.i1 = nn.Conv2d(self.channel+self.input_addition, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.conv = nn.Conv2d(self.channel+self.input_addition, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            
        if 'ad2'in mode:
            self.app = nn.Conv2d(self.channel+self.input_addition, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            self.dis = nn.Conv2d(self.channel+self.input_addition, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        else:
            self.res = nn.Conv2d(self.channel+self.input_addition, self.channel, kernel_size, padding=padding, stride=1, bias=False)
            
        self.exchange = nn.Conv2d(self.channel+self.input_addition, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        if 'c' in mode:
            self.concat = nn.Conv2d((self.channel+self.input_addition)*2, self.channel, kernel_size, padding=padding, stride=1, bias=False)
        print("MTF: mode: {} kernel_size: {}".format(self.mode, kernel_size))
        
    def forward(self, f0, f1):
        #t0 = self.conv(f0)
        #t1 = self.conv(f1)

        if 'i2' in self.mode:
            info = self.i0(f0) + self.i1(f1)
        else:
            info = self.conv(f0 + f1)
            
        if 'd' in self.mode:
            if 'ad2' in self.mode:
                disappear = self.dis(self.relu(f0 - f1))
            else:
                disappear = self.res(self.relu(f0 - f1))
        else:
            disappear = 0

        if 'a' in self.mode:
            if 'ad2' in self.mode:
                appear = self.app(self.relu(f1 - f0))
            else:
                appear = self.res(self.relu(f1 - f0))
        else:
            appear = 0

        if 'e' in self.mode:
            exchange = self.exchange(torch.max(f0, f1) - torch.min(f0, f1))
        else:
            exchange = 0

        if 'c' in self.mode:
            concat = self.concat(torch.cat((f0, f1), dim=1))
        else:
            concat = 0

        if self.mode == 'i':
            f = info
        elif self.mode == 'a':
            f = appear
        elif self.mode == 'd':
            f = disappear
        elif self.mode == 'e':
            f = exchange
        elif self.mode == 'ia':
            f = info + 2 * appear
        elif self.mode in ['id', 'i2d']:
            f = info + 2 * disappear
        elif self.mode == 'ie':
            f = info + 2 * exchange
        elif self.mode == 'iae':
            f = info + appear + exchange
        elif self.mode == 'ide':
            f = info + disappear + exchange
        elif self.mode == 'iad':
            f = info + disappear + appear
        elif self.mode in ['iade', 'i2ade', 'iad2e', 'i2ad2e']:
            f = info + disappear + appear + exchange
        elif self.mode == 'iadec':
            f = info + disappear + appear + exchange + concat

        f = self.relu(f)
        return f


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.deconv1 = UpsampleConvLayer(in_channels, in_channels, kernel_size=4, stride=2)
        self.deconv2 = UpsampleConvLayer(in_channels, in_channels, kernel_size=4, stride=2)
        self.classifier = nn.Conv2d(in_channels, out_channels, 1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.deconv1(x)
        x = self.bn(x)
        # x = self.deconv2(x)
        x = self.relu(x)
        pred = self.classifier(x)

        return pred


def efficientnet_ayanet():
    channels = [32, 48, 80, 160]
    model = efficientnet_ayn(weights=None, input_channel=3)
                
    return model, channels

def gabor_encoder():
    input_channels=8
    channels = [32, 48, 80, 160]
    model = GaborEncoder(input_channels=input_channels, channels=channels)

    return model, channels

def get_encoder(arch,pretrained=True):
    if arch == 'gaborencoder':
        return gabor_encoder()
    elif arch == 'efficientnet_ayn':
        return efficientnet_ayanet()
    else:
        print('Invalid architecture for encoder...')
        exit(-1)

def get_mtf_module(mode='iade', kernel_size=1, channels = [64,128,256,512], input_addition=0):
    # return MTFModule(mode=mode, kernel_size=kernel_size, channels=channels, input_addition=input_addition)
    return MTFModule2(mode=mode, kernel_size=kernel_size, channels=channels, input_addition=input_addition)
    # return MTFModule3(mode=mode, kernel_size=kernel_size, channels=channels, input_addition=input_addition)

def get_decoder(arch,channels=[64,128,256,512]):
    if arch == 'ayanet':
        return Decoder(channels=channels)
        # return Decoder2(channels=channels)
    else:
        print('Invalid architecture for decoder...')
        exit(-1)

def get_classifier(in_channels, out_channels):
    return Classifier(in_channels=in_channels, out_channels=out_channels)

