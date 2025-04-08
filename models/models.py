'''
The implementation of ADA-Net: Attention-Guided Domain Adaptation Network with Contrastive Learning for Standing Dead Tree Segmentation Using Aerial Imagery.
Author: Mete Ahishali,

The software implementation is extensively based on the following repository: https://github.com/taesungp/contrastive-unpaired-translation.
'''
import torch
import numpy as np

from torch import nn
from math import log

import models.blocks as blocks

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, padding='same', padding_mode='reflect', kernel_size=3)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, padding='same', padding_mode='reflect', kernel_size=3)
        self.value_conv = nn.Conv2d(in_dim, in_dim, padding='same', padding_mode='reflect', kernel_size=3)
        
        # Learnable parameter to scale the output.
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Compute the attention map.
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention to the value matrix.
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Scale the output and use skip connection.
        out = self.gamma * out + x
        return out


class SkipConnect(nn.Module):

    def __init__(self, dim):

        super(SkipConnect, self).__init__()

        resnet_block = []
        resnet_block += [nn.ReflectionPad2d(1)]
        resnet_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=False),  nn.InstanceNorm2d(dim), nn.ReLU(True)]
        resnet_block += [nn.ReflectionPad2d(1)]
        resnet_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=False), nn.InstanceNorm2d(dim)]
        self.resnet_block = nn.Sequential(*resnet_block)

    def forward(self, x):
        out = x + self.resnet_block(x) 
        return out

class generator(nn.Module):

    def __init__(self, input_channels=4, output_channels=4):

        super(generator, self).__init__()

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channels, 64, kernel_size=7, padding=0, bias=False),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(True)]

        model += [nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(128),
                    nn.ReLU(True)]

        model += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(256),
                    nn.ReLU(True)]
            
        
        model += [SkipConnect(256)]
        model += [SelfAttention(256), nn.InstanceNorm2d(256)]
        model += [SkipConnect(256)]
        model += [SkipConnect(256)]
        model += [SelfAttention(256), nn.InstanceNorm2d(256)]
        model += [SkipConnect(256)]

        
        model += [nn.ConvTranspose2d(256, 128,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=False),
                    nn.InstanceNorm2d(128),
                    nn.ReLU(True)]

        model += [nn.ConvTranspose2d(128, 64,
                                kernel_size=3, stride=2,
                                padding=1, output_padding=1,
                                bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)]            
        
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(64, output_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def forward(self, input, layers=[]):
        if len(layers) > 0:
            x = input
            features = []
            for i, layer in enumerate(self.model):
                x = layer(x)
                if i in layers:
                    features.append(x)
                else:
                    pass
                if i == layers[-1]:
                    return features
        else:
            fake = self.model(input)
            return fake


class MLPs(nn.Module):
    def __init__(self, feature_dimensions):

        super(MLPs, self).__init__()

        self.num_patches = 256

        self.MLPs = nn.ModuleList()

        for i in range(0, len(feature_dimensions)):
            self.MLPs.append(nn.Sequential(*[nn.Linear(feature_dimensions[i], 256), nn.ReLU(), nn.Linear(256, 256)]))

            if torch.cuda.is_available():
                self.MLPs[i] = self.MLPs[i].to('cuda')


    def forward(self, features, sample_ids=None):
        return_ids = []
        return_features = []

        for i in range(0, len(features)):

            feat_reshape = features[i].permute(0, 2, 3, 1).flatten(1, 2)

            if sample_ids is not None:
                sample_id = sample_ids[i]
            else:
                sample_id = np.random.permutation(feat_reshape.shape[1])
                sample_id = sample_id[:int(min(self.num_patches, sample_id.shape[0]))]
            sample_id = torch.tensor(sample_id, dtype=torch.long, device=features[i].device)
            x_sample = feat_reshape[:, sample_id, :].flatten(0, 1)

            x_sample = self.MLPs[i](x_sample)
            return_ids.append(sample_id)

            norm = x_sample.pow(2).sum(1, keepdim=True).pow(1. / 2)
            x_sample = x_sample.div(norm + 1e-7)

            return_features.append(x_sample)
        return return_features, return_ids

class freq_patch(nn.Module):
    def __init__(self, input_channels, freq_patch_size, freq_num_patches):
        super(freq_patch, self).__init__()

        self.freq_patch_size = freq_patch_size
        self.freq_num_patches = freq_num_patches
        
        input_dim = 2 * input_channels * freq_patch_size * freq_patch_size
        self.mlp = nn.Sequential(*[nn.Linear(input_dim, 1024), nn.ReLU(), nn.Linear(1024, 256)])

        if torch.cuda.is_available():
            self.mlp = self.mlp.to('cuda')
            self.normalize = nn.BatchNorm1d(input_dim, device = 'cuda')
        else:
            self.normalize = nn.BatchNorm1d(input_dim)

    def forward(self, feats, patch_ids=None):
        return_ids = []
        return_feats = []

        B, H, W = feats.shape[0], feats.shape[2], feats.shape[3]

        for i in range(0, self.freq_num_patches):
            if patch_ids is not None:
                start_id_x, start_id_y = patch_ids[i]
            else:
                start_id_x = np.random.permutation(H - self.freq_patch_size)[0]
                start_id_y = np.random.permutation(W - self.freq_patch_size)[0]
                return_ids.append((start_id_x, start_id_y))

            x_sample = feats[:, :,
                            start_id_x : start_id_x + self.freq_patch_size,
                            start_id_y : start_id_y + self.freq_patch_size]

            freq = torch.fft.fft2(x_sample, norm='ortho')
            x_sample = torch.stack([freq.real, freq.imag], -1)

            x_sample = x_sample.flatten(1, -1)

            x_sample = self.normalize(x_sample)
            
            x_sample = self.mlp(x_sample)

            norm = x_sample.pow(2).sum(1, keepdim=True).pow(1. / 2)
            x_sample = x_sample.div(norm + 1e-7)

            return_feats.append(x_sample)                

        return return_feats, return_ids


class discriminator(nn.Module):
    def __init__(self, output_channels, size=None, load_size=256):
        super().__init__()
        self.stddev_group = 16
        size = 2 ** int((np.rint(np.log2(load_size))))

        blur_kernel = [1, 3, 3, 1]

        channels = {
            4: min(384, 4096),
            8: min(384, 2048),
            16: min(384, 1024),
            32: min(384, 512),
            64: 256,
            128: 128,
            256: 64,
            512: 32,
            1024: 16,
        }

        convs = [blocks.ConvLayer(output_channels, channels[size], 1)]

        log_size = int(log(size, 2))
        in_channel = channels[size]
        final_res_log2 = 2

        for i in range(log_size, final_res_log2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(blocks.ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = blocks.ConvLayer(in_channel, channels[4], 3)
        

        self.final_linear = nn.Sequential(
            blocks.EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            blocks.EqualLinear(channels[4], 1),
        )

        if torch.cuda.is_available():
            self.convs = self.convs.to('cuda')
            self.final_conv = self.final_conv.to('cuda')
            self.confinal_linearvs = self.final_linear.to('cuda')


    def forward(self, input):
        out = input
        for conv in self.convs:
            out = conv(out)


        batch, _, _, _ = out.shape

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out