import os
import torch
import torch as T
import torch.optim as optim
from torchvision.transforms import functional as F
from neural_monitor import monitor as mon
from neural_monitor import logger
import argparse

from param_stroke import BrushStrokeRenderer
import utils
import losses

parser = argparse.ArgumentParser()
parser.add_argument('content_img_file', type=str, help='Content image file')
parser.add_argument('style_img_file', type=str, help='Style image file')
parser.add_argument('--img_size', '-s', type=int, default=512,
                    help='The smaller dimension of content image is resized into this size. Default: 512.')
parser.add_argument('--canvas_color', default='gray', type=str,
                    help='Canvas background color (`gray` (default), `white`, `black` or `noise`).')
parser.add_argument('--num_strokes', default=5000, type=int,
                    help='Number of strokes to draw. Default: 5000.')
parser.add_argument('--samples_per_curve', default=10, type=int,
                    help='Number of points to sample per parametrized curve. Default: 10.')
parser.add_argument('--brushes_per_pixel', default=20, type=int,
                    help='Number of brush strokes to be drawn per pixel. Default: 20.')
parser.add_argument('--output_path', '-o', type=str, default='results',
                    help='Storage for results. Default: `results`.')
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help='Device to perform stylization. Default: `cuda`.')
args = parser.parse_args()

# inputs
style_img_file = args.style_img_file
content_img_file = args.content_img_file

# setup logging
model_name = 'nst-stroke'
root = args.output_path
vgg_weight_file = 'vgg_weights/vgg19_weights_normalized.h5'
print_freq = 10
mon.initialize(model_name=model_name, root=root, print_freq=print_freq)
mon.backup(('main.py', 'param_stroke.py', 'utils.py', 'losses.py', 'vgg.py'))

# device
device = torch.device(args.device)

# desired size of the output image
imsize = args.img_size
content_img = utils.image_loader(content_img_file, imsize, device)
style_img = utils.image_loader(style_img_file, 224, device)
output_name = f'{os.path.basename(content_img_file).split(".")[0]}-{os.path.basename(style_img_file).split(".")[0]}'

# desired depth layers to compute style/content losses :
bs_content_layers = ['conv4_1', 'conv5_1']
bs_style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
px_content_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
px_style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# brush strokes parameters
canvas_color = args.canvas_color
num_strokes = args.num_strokes
samples_per_curve = args.samples_per_curve
brushes_per_pixel = args.brushes_per_pixel
_, _, H, W = content_img.shape
canvas_height = H
canvas_width = W
length_scale = 1.1
width_scale = 0.1


def run_stroke_style_transfer(num_steps=100, style_weight=3., content_weight=1., tv_weight=0.008, curv_weight=4):
    vgg_loss = losses.StyleTranferLosses(vgg_weight_file, content_img, style_img,
                                         bs_content_layers, bs_style_layers, scale_by_y=True)
    vgg_loss.to(device).eval()

    # brush stroke init
    bs_renderer = BrushStrokeRenderer(canvas_height, canvas_width, num_strokes, samples_per_curve, brushes_per_pixel,
                                      canvas_color, length_scale, width_scale,
                                      content_img=content_img[0].permute(1, 2, 0).cpu().numpy())
    bs_renderer.to(device)

    optimizer = optim.Adam([bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                            bs_renderer.curve_c, bs_renderer.width], lr=1e-1)
    optimizer_color = optim.Adam([bs_renderer.color], lr=1e-2)

    logger.info('Optimizing brushstroke-styled canvas..')
    for _ in mon.iter_batch(range(num_steps)):
        optimizer.zero_grad()
        optimizer_color.zero_grad()
        input_img = bs_renderer()
        input_img = input_img[None].permute(0, 3, 1, 2).contiguous()
        content_score, style_score = vgg_loss(input_img)

        style_score *= style_weight
        content_score *= content_weight
        tv_score = tv_weight * losses.total_variation_loss(bs_renderer.location, bs_renderer.curve_s,
                                                           bs_renderer.curve_e, K=10)
        curv_score = curv_weight * losses.curvature_loss(bs_renderer.curve_s, bs_renderer.curve_e, bs_renderer.curve_c)
        loss = style_score + content_score + tv_score + curv_score
        loss.backward(inputs=[bs_renderer.location, bs_renderer.curve_s, bs_renderer.curve_e,
                              bs_renderer.curve_c, bs_renderer.width], retain_graph=True)
        optimizer.step()
        style_score.backward(inputs=[bs_renderer.color])
        optimizer_color.step()

        # plot some stuffs
        mon.plot('stroke style loss', style_score.item())
        mon.plot('stroke content loss', content_score.item())
        mon.plot('stroke tv loss', tv_score.item())
        mon.plot('stroke curvature loss', curv_score.item())
        if mon.iter % mon.print_freq == 0:
            mon.imwrite('stroke stylized', input_img)

    with T.no_grad():
        return bs_renderer()


def run_style_transfer(input_img: T.Tensor, num_steps=1000, style_weight=10000., content_weight=1., tv_weight=0.):
    content_img_resized = F.resize(content_img, 1024)

    input_img = input_img.detach()[None].permute(0, 3, 1, 2).contiguous()
    input_img = F.resize(input_img, 1024)
    input_img = T.nn.Parameter(input_img, requires_grad=True)

    vgg_loss = losses.StyleTranferLosses(vgg_weight_file, content_img_resized, style_img,
                                         px_content_layers, px_style_layers)
    vgg_loss.to(device).eval()
    optimizer = optim.Adam([input_img], lr=1e-3)
    logger.info('Optimizing pixel-wise canvas..')
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


if __name__ == '__main__':
    # optimize brush style transfer model
    canvas = run_stroke_style_transfer()

    # optimize the canvas pixel-wise
    mon.iter = 0
    mon.print_freq = 100
    output = run_style_transfer(canvas)
    mon.imwrite(output_name, output)
    logger.info('Finished!')
