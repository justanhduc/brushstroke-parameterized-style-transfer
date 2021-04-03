import torch as T
from imageio import imwrite
from torch.nn import functional as F
from neuralnet_pytorch import monitor as mon


def bezier_curve(p0: T.Tensor, p1: T.Tensor, p2: T.Tensor, t: T.Tensor):
    """
    p0, p1, p2: Nx2
    t: NxS
    :param p0:
    :param p1:
    :param p2:
    :param t:
    :return:
    """
    t = t[..., None]
    p0, p1, p2 = p0[:, None], p1[:, None], p2[:, None]
    return (1. - t ** 2.) * p0 + 2 * (1. - t) * t * p1 + t ** 2 * p2


def stroke_renderer(input: T.Tensor, stroke_params: T.Tensor, n_samples=10, temp=8., img_size=256, k=20):
    """

    :param stroke_params: Nx12, \in [0, 1]
    :param temp: float
    :param n_samples: int
    :return:
    """
    # unpack parameters
    p0, p1, p2, stroke_widths, stroke_color = T.split(stroke_params, [2, 2, 2, 1, 3], dim=-1)
    stroke_color = T.tanh(stroke_color) / 2. + .5  # need it later
    stroke_widths = T.exp(stroke_widths) / img_size  # no negative, need it later

    # setup coordinates
    coords = T.linspace(0, 1, img_size).to(stroke_params.device)
    coords = T.stack(T.meshgrid((coords, coords)), dim=2)
    new_img_size = int(.1 * img_size)
    # coords_coarse = T.linspace(0, 1, new_img_size).to(stroke_params.device)
    # coords_coarse = T.stack(T.meshgrid((coords_coarse, coords_coarse)), dim=2)
    coords_coarse = F.interpolate(coords.permute(2, 0, 1)[None],  # coarse coordinates H'xW'x2
                                  (new_img_size, new_img_size), mode='bilinear',
                                  align_corners=False)[0].permute(1, 2, 0).contiguous()
    strokes_pos = T.stack((p0, p1, p2), dim=1)

    # distance tensor
    dists = T.min(T.sum((coords_coarse[:, :, None, None] - strokes_pos[None, None]) ** 2, dim=-1), dim=-1)[0]  # H'xW'xN
    nearest_stroke_indices_ = T.topk(dists, k, largest=False, dim=-1)[1]  # take the nearest k stroke indices
    nearest_stroke_indices = F.interpolate(nearest_stroke_indices_.float().permute(2, 0, 1)[None],  # interpolate with nearest neighbor
                                           (img_size, img_size), mode='nearest')[0].permute(1, 2, 0).contiguous().long()
    # dists = T.min(T.sum((coords[:, :, None, None] - strokes_pos[None, None]) ** 2, dim=-1), dim=-1)[0]  # HxWxN
    # nearest_stroke_indices = T.topk(dists, k, largest=False, dim=-1)[1]  # take the nearest k stroke indices

    # sample equi-distant points on the curves
    samples = T.linspace(0, 1, n_samples).to(stroke_params.device)[None]
    samples = bezier_curve(p0, p1, p2, samples)
    samples = samples[nearest_stroke_indices.flatten()].view(img_size, img_size, k, n_samples, -1)

    dists = T.sqrt(T.sum((coords[:, :, None, None] - samples) ** 2, dim=-1) + 1e-8)  # HxWxNxS
    d_strokes = T.min(dists, dim=-1)[0]

    # stroke masks
    swidths = stroke_widths[nearest_stroke_indices.flatten()].view(img_size, img_size, k)
    m_strokes = T.sigmoid(temp * (swidths - d_strokes))  # HxWxN

    # render strokes
    # scolors = stroke_color[nearest_stroke_indices_.flatten()].view(new_img_size, new_img_size, k, 3)
    # scolors = F.interpolate(scolors.permute(2, 3, 0, 1), [img_size, img_size],
    #                         mode='bilinear', align_corners=False).permute(2, 3, 0, 1).contiguous()
    # I_strokes = m_strokes[..., None] * scolors
    I_strokes = m_strokes[..., None] * stroke_color[nearest_stroke_indices.flatten()].view(img_size, img_size, k, 3)

    # aggregate all strokes
    A = T.softmax(-temp * d_strokes, dim=-1)
    I = T.sum(I_strokes * A[..., None], dim=-2)
    return I  # HxWx3


if __name__ == '__main__':
    import math

    strokes = T.tensor([
        [.5, .5, .6, .2, .7, .3, math.log(10.), 0., 0., 1.],
        [.1, .2, .102, .14, .21, .3, math.log(20.), 1., 0., 0.],
        [.8, .2, .6, .3, .9, .35, math.log(5.), .0, 1, 0.]
    ])
    img = stroke_renderer(None, strokes, temp=500., k=2)
    imwrite('strokes.jpg', img.detach().numpy())
