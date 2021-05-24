import os
import torch
import torch as T
import torch.optim as optim
import numpy as np
from torchvision.transforms import functional as F
from neural_monitor import monitor as mon
from neural_monitor import logger

from param_stroke import stroke_renderer, sample_quadratic_bezier_curve
import utils
import losses

# inputs
style_img_file = "/ssd2/duc/stroke_nst/images/girl-on-a-divan.jpg"
content_img_file = "/ssd2/duc/stroke_nst/images/elefant.jpg"

# setup logging
model_name = 'nst-stroke'
root = '/ssd2/duc/stroke_nst/results'
vgg_weight_file = '/ssd2/duc/stroke_nst/vgg_weights/vgg19_weights_normalized.h5'
print_freq = 10
mon.initialize(model_name=model_name, root=root, print_freq=print_freq)
mon.backup(('nst.py', 'param_stroke.py', 'utils.py'))

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
content_img = utils.image_loader(content_img_file, imsize, device)
style_img = utils.image_loader(style_img_file, 224, device)
output_name = f'{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}'

# desired depth layers to compute style/content losses :
bs_content_layers = ['conv4_1', 'conv5_1']
bs_style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
px_content_layers = ['conv3_1', 'conv4_1', 'conv5_1']
px_style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# brush strokes parameters
canvas_color = .5
num_strokes = 5000
S = 10
K = 20
_, _, H, W = content_img.shape
canvas_height = H
canvas_width = W
length_scale = 1.1
width_scale = 0.1

# brush stroke init
location, s, e, c, width, color = utils.initialize_brushstrokes(content_img[0].permute(1, 2, 0).cpu().numpy(),
                                                                num_strokes,
                                                                canvas_height,
                                                                canvas_width,
                                                                length_scale,
                                                                width_scale)
location = location[..., ::-1]
s = s[..., ::-1]
e = e[..., ::-1]
c = c[..., ::-1]
curve_s = T.nn.Parameter(T.from_numpy(np.array(s, 'float32')).to(device), requires_grad=True)
curve_e = T.nn.Parameter(T.from_numpy(np.array(e, 'float32')).to(device), requires_grad=True)
curve_c = T.nn.Parameter(T.from_numpy(np.array(c, 'float32')).to(device), requires_grad=True)
color = T.nn.Parameter(T.from_numpy(color).to(device), requires_grad=True)
location = T.nn.Parameter(T.from_numpy(np.array(location, 'float32')).to(device), requires_grad=True)
width = T.nn.Parameter(T.from_numpy(width).to(device), requires_grad=True)


def get_stroke_optimizer():
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([location, curve_s, curve_e, curve_c, width], lr=1e-1)
    optimizer_color = optim.Adam([color], lr=1e-2)
    return optimizer, optimizer_color


def get_input_optimizer(input_image):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.Adam([input_image], lr=1e-3)
    return optimizer


def run_stroke_style_transfer(num_steps=100, style_weight=3., content_weight=1., tv_weight=0.008, curv_weight=4):
    logger.info('Optimizing brushstroke-styled canvas..')
    vgg_loss = losses.StyleTranferLosses(vgg_weight_file, content_img, style_img,
                                         bs_content_layers, bs_style_layers, scale_by_y=True)
    vgg_loss.to(device).eval()
    optimizer, optimizer_color = get_stroke_optimizer()
    for _ in mon.iter_batch(range(num_steps)):
        optimizer.zero_grad()
        optimizer_color.zero_grad()
        curve_points = sample_quadratic_bezier_curve(s=curve_s + location,
                                                     e=curve_e + location,
                                                     c=curve_c + location,
                                                     num_points=S)
        input_img = stroke_renderer(curve_points, location, color, width, canvas_height, canvas_width, K, canvas_color)
        input_img = input_img[None].permute(0, 3, 1, 2).contiguous()
        content_score, style_score = vgg_loss(input_img)

        style_score *= style_weight
        content_score *= content_weight
        tv_score = tv_weight * losses.total_variation_loss(location, curve_s, curve_e, K=10)
        curv_score = curv_weight * losses.curvature_loss(curve_s, curve_e, curve_c)
        loss = style_score + content_score + tv_score + curv_score
        loss.backward(inputs=[location, curve_s, curve_e, curve_c, width], retain_graph=True)
        optimizer.step()
        style_score.backward(inputs=[color])
        optimizer_color.step()

        # plot some stuffs
        mon.plot('stroke style loss', style_score.item())
        mon.plot('stroke content loss', content_score.item())
        mon.plot('stroke tv loss', tv_score.item())
        mon.plot('stroke curvature loss', curv_score.item())
        if mon.iter % mon.print_freq == 0:
            mon.imwrite('stroke stylized', input_img)


run_stroke_style_transfer()


def run_style_transfer(num_steps=1000, style_weight=10000., content_weight=1., tv_weight=0.):
    logger.info('Optimizing pixel-wise canvas..')
    content_img_resized = F.resize(content_img, 1024)

    with T.no_grad():
        curve_points = sample_quadratic_bezier_curve(s=curve_s + location, e=curve_e + location,
                                                     c=curve_c + location, num_points=S)
        input_img = stroke_renderer(curve_points, location, color, width, canvas_height, canvas_width, K, canvas_color)

    input_img = input_img.detach()[None].permute(0, 3, 1, 2).contiguous()
    input_img = F.resize(input_img, 1024)
    input_img = T.nn.Parameter(input_img, requires_grad=True)

    vgg_loss = losses.StyleTranferLosses(vgg_weight_file, content_img_resized, style_img,
                                         px_content_layers, px_style_layers)
    vgg_loss.to(device).eval()
    optimizer = get_input_optimizer(input_img)
    for _ in mon.iter_batch(range(num_steps)):
        optimizer.zero_grad()
        input = T.clamp(input_img, 0., 1.)
        content_score, style_score = vgg_loss(input)

        style_score *= style_weight
        content_score *= content_weight
        tv_score = 0. if not tv_weight else tv_weight * losses.tv_loss(input_img)
        loss = style_score + content_score + tv_score
        loss.backward(inputs=[input_img])
        optimizer.step()

        # plot some stuffs
        mon.plot('pixel style loss', style_score)
        mon.plot('pixel content loss', content_score)
        mon.plot('pixel tv loss', tv_score)
        if mon.iter % mon.print_freq == 0:
            mon.imwrite('pixel stylized', input)

    return T.clamp(input_img, 0., 1.)


mon.iter = 0
mon.print_freq = 100
output = run_style_transfer()
mon.imwrite(output_name, output)
logger.info('Finished!')
