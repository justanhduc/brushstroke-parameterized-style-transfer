import torch as T
import torch.nn as nn
import torch.nn.init as init
from neural_monitor import logger
import h5py
import numpy as np
import torch.nn.functional as F

import utils


class Normalization(T.nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        self.register_buffer('kern', T.from_numpy(np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], 'float32')[:, :, None, None]))
        self.register_buffer('bias', T.from_numpy(np.array([-103.939, -116.779, -123.68], 'float32')))

    def forward(self, input):
        return F.conv2d(input, self.kern, bias=self.bias, padding=0)


class ConvRelu(T.nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias: bool = True,
                 padding_mode: str = 'zeros'  # TODO: refine this type
                 ):
        super(ConvRelu, self).__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU()


class VGG19(T.nn.Sequential):
    def __init__(self, weight_file):
        super(VGG19, self).__init__()
        self.norm = Normalization()

        self.conv1_1 = ConvRelu(3, 64)
        self.conv1_2 = ConvRelu(64, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv2_1 = ConvRelu(64, 128)
        self.conv2_2 = ConvRelu(128, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv3_1 = ConvRelu(128, 256)
        self.conv3_2 = ConvRelu(256, 256)
        self.conv3_3 = ConvRelu(256, 256)
        self.conv3_4 = ConvRelu(256, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv4_1 = ConvRelu(256, 512)
        self.conv4_2 = ConvRelu(512, 512)
        self.conv4_3 = ConvRelu(512, 512)
        self.conv4_4 = ConvRelu(512, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv5_1 = ConvRelu(512, 512)
        self.conv5_2 = ConvRelu(512, 512)
        self.conv5_3 = ConvRelu(512, 512)
        self.conv5_4 = ConvRelu(512, 512)
        self.pool5 = nn.MaxPool2d(2, stride=2)
        self.load_params(weight_file)

    def load_params(self, param_file):
        if param_file is not None:
            f = h5py.File(param_file, mode='r')
            trained = [np.array(layer[1], 'float32') for layer in list(f.items())]
            weight_value_tuples = []
            for p, tp in zip(self.parameters(), trained):
                if len(tp.shape) == 4:
                    tp = np.transpose(tp, (3, 2, 0, 1))
                weight_value_tuples.append((p, tp))
            utils.batch_set_value(*zip(*(weight_value_tuples)))
            logger.info('Pretrained weights loaded successfully!')
