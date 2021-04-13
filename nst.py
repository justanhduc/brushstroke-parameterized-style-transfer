from __future__ import print_function

import os
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import functional as TF

import copy
from neuralnet_pytorch import monitor as mon

from param_stroke import stroke_renderer

mon.model_name = 'nst-stroke'
mon.root = '/ssd2/duc/stroke_nst/results'
mon.set_path()
mon.backup(('nst.py', 'param_stroke.py'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name, return_aspect_ratio=False):
    image = Image.open(image_name)
    if return_aspect_ratio:
        w, h = image.size
        aspect_ratio = w / h

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    image = image.to(device, torch.float)
    return image if not return_aspect_ratio else (image, aspect_ratio)


style_img = image_loader("/ssd2/duc/stroke_nst/images/girl-on-a-divan.jpg")
content_img, aspect_ratio = image_loader("/ssd2/duc/stroke_nst/images/golden-gate-bridge.jpg", True)


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# input_img = content_img.clone()
num_strokes = 5000
c = T.nn.Parameter(T.rand(num_strokes, 2).to(device), requires_grad=True)
p0 = T.nn.Parameter(T.rand(num_strokes, 2).to(device) * .1, requires_grad=True)
p1 = T.nn.Parameter(T.rand(num_strokes, 2).to(device) * .1, requires_grad=True)
p2 = T.nn.Parameter(T.rand(num_strokes, 2).to(device) * .1, requires_grad=True)
swidths = T.nn.Parameter(T.log(T.ones(num_strokes, 1, device=device) * 20.), requires_grad=True)
scolors = T.rand(num_strokes, 3, requires_grad=True, device=device)


def get_stroke_optimizer():
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([c, p0, p1, p2, swidths, scolors], lr=3e-3)
    return optimizer


def get_input_optimizer(input_image):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([input_image.requires_grad_(True)], lr=1e-3)
    return optimizer


def tv_loss(x):
    diff_i = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    diff_j = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    loss = diff_i + diff_j
    return loss


def run_stroke_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, num_steps=1000,
                              style_weight=1e4, content_weight=1, tv_weight=1.):
    """Run the style transfer."""
    print('Building the brushstroke style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                     style_img, content_img)
    optimizer = get_stroke_optimizer()

    print('Optimizing..')
    for step in mon.iter_batch(range(1, num_steps + 1)):
        def closure():
            # correct the values of updated input image
            # input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            input_img = stroke_renderer(content_img, T.cat((c, p0, p1, p2, swidths, scolors), dim=-1), temp=150.,
                                        img_size=imsize, k=20)
            input_img = input_img[None].permute(0, 3, 1, 2).contiguous()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            tv_score = tv_weight * tv_loss(input_img)
            loss = style_score + content_score + tv_score
            loss.backward()

            mon.plot('stroke style loss', style_score.item())
            mon.plot('stroke content loss', content_score.item())
            mon.plot('stroke tv loss', tv_score.item())
            if step % 100 == 0:
                mon.imwrite('stroke stylized', input_img)

            return style_score + content_score

        optimizer.step(closure)


run_stroke_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img)
# input_img = stroke_renderer(content_img, T.cat((p0, p1, p2, swidths, scolors), dim=-1), temp=150.,
#                             img_size=imsize, k=20)
# from imageio import imread
# input_img = imread('/ssd2/duc/stroke_nst/results/nst-stroke/run-21/images/stylized_1000_0.jpg')
# input_img = T.from_numpy(input_img).to(device) / 255.


def run_neural_style_transfer(cnn, normalization_mean, normalization_std, content_img, style_img, input_img,
                              num_steps=1000, style_weight=1e5, content_weight=1, tv_weight=1.):
    """Run the style transfer."""
    print('Building the neural style transfer model..')
    # resize image to 1024 and keep aspect ratio
    if aspect_ratio < 1:
        w = 1024
        new_img_size = (int(w / aspect_ratio), w)
    else:
        h = 1024
        new_img_size = (h, int(h * aspect_ratio))

    input_img = input_img.detach()[None].permute(0, 3, 1, 2).contiguous()
    input_img = F.interpolate(input_img, new_img_size, mode='bilinear', align_corners=True)
    content_img = F.interpolate(content_img, new_img_size, mode='bilinear', align_corners=True)
    style_img = F.interpolate(style_img, new_img_size, mode='bilinear', align_corners=True)
    model, style_losses, content_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                     style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    for step in mon.iter_batch(range(1, num_steps + 1)):
        def closure():
            # correct the values of updated input image
            # input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight
            tv_score = tv_weight * tv_loss(input_img)
            loss = style_score + content_score + tv_score
            loss.backward()

            mon.plot('style loss', style_score.item())
            mon.plot('content loss', content_score.item())
            mon.plot('tv loss', tv_score.item())
            if step % 100 == 0:
                mon.imwrite('stylized', input_img)

            return style_score + content_score

        optimizer.step(closure)


# run_neural_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)
print('Finished!')
