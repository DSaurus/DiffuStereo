import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lib.network.ImageEncoder.net_util import ConvBlock


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch', base=64):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm
        self.base = base

        self._generate_network(self.depth)

    def _generate_network(self, level):
        base = self.base
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features, norm=self.norm, base=base))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features, norm=self.norm, base=base))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features, norm=self.norm, base=base))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features, norm=self.norm, base=base))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3, scale_factor=2, mode='bicubic', align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Module):
    def __init__(self, opt):
        super(HGFilter, self).__init__()
        self.num_modules = opt.num_stack

        self.opt = opt
        self.downsample = opt.downsample
        base = opt.base
        # Base part
        if self.downsample == 0:
            self.conv1_h = nn.Conv2d(3, base, kernel_size=3, stride=1, padding=1)
            self.conv1_nh = nn.Conv2d(6, base, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, base, kernel_size=7, stride=2, padding=3)
            self.conv1_n = nn.Conv2d(6, base, kernel_size=7, stride=2, padding=3)
        
        if self.opt.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(base)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(base // 2, base)

        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(base, base, self.opt.norm, base)
            self.down_conv2 = nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(base, base * 2, self.opt.norm, base)
            self.down_conv2 = nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=2, padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(base, base * 2, self.opt.norm, base)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(base * 2, base * 2, self.opt.norm, base)
        self.conv4 = ConvBlock(base * 2, base * 4, self.opt.norm, base)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, opt.num_hourglass, base * 4, self.opt.norm, base=base))

            self.add_module('top_m_' + str(hg_module), ConvBlock(base * 4, base * 4, self.opt.norm, base))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(base * 4, base * 4, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(base * 4))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module), nn.GroupNorm(base // 2, base * 4))
                
            self.add_module('l' + str(hg_module), nn.Conv2d(base * 4,
                                                            opt.hourglass_dim, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(base * 4, base * 4, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(opt.hourglass_dim,
                                                                 base * 4, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        if self.opt.input_smpl_normal:
            if self.downsample == 0:
                x = F.relu(self.bn1(self.conv1_nh(x)), True)
            else:    
                x = F.relu(self.bn1(self.conv1_n(x)), True)
        else:
            if self.downsample == 0:
                x = F.relu(self.bn1(self.conv1_h(x)), True)
            else:
                x = F.relu(self.bn1(self.conv1(x)), True)

        tmpx = x
        if self.opt.hg_down == 'ave_pool':
            if self.downsample == 0:
                x = self.conv2(x)
            else:
                x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs, tmpx.detach(), normx

class MVHGFilter(nn.Module):
    def __init__(self, opt):
        super(MVHGFilter, self).__init__()
        self.HGFilter = HGFilter(opt)
    
    def forward(self, x):
        B, V, C, H, W = x.shape
        x = rearrange(x, 'b v c h w -> (b v) c h w')
        output, _, _ = self.HGFilter(x)
        output = rearrange(output[-1], '(b v) c h w -> b v c h w', b = B)
        return output

