import torch as T
from imageio import imwrite
from torch.nn import functional as F
from torch_cluster import knn


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
    return (1. - t) ** 2. * p0 + 2 * (1. - t) * t * p1 + t ** 2 * p2


def stroke_renderer(input: T.Tensor, stroke_params: T.Tensor, n_samples=10, temp=8., img_size=256, k=20):
    """

    :param stroke_params: Nx12, \in [0, 1]
    :param temp: float
    :param n_samples: int
    :return:
    """
    # unpack parameters
    c, p0, p1, p2, stroke_widths, stroke_color = T.split(stroke_params, [2, 2, 2, 2, 1, 3], dim=-1)
    p0 = T.clamp(c + p0, 0., 1.)
    p1 = T.clamp(c + p1, 0., 1.)
    p2 = T.clamp(c + p2, 0., 1.)
    stroke_color = T.clamp(stroke_color, 0., 1.)  # need it later
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
    # strokes_pos = T.stack((p0, p1, p2), dim=1)

    # distance tensor
    nearest_stroke_indices = knn(c.view(-1, 2), coords_coarse.view(-1, 2), k)[1].view(new_img_size, new_img_size, k)
    # nearest_stroke_indices_ = nearest_stroke_indices_[1].view(new_img_size, new_img_size, k) // 3
    nearest_stroke_indices = F.interpolate(nearest_stroke_indices.float().permute(2, 0, 1)[None],  # interpolate with nearest neighbor
                                           (img_size, img_size), mode='nearest')[0].permute(1, 2, 0).contiguous().long()
    # nearest_stroke_indices = knn(strokes_pos.view(-1, 2), coords.view(-1, 2), k)
    # nearest_stroke_indices = nearest_stroke_indices[1].view(img_size, img_size, k) // 3

    # sample equi-distant points on the curves
    samples = T.linspace(0, 1, n_samples).to(stroke_params.device)[None]
    samples = bezier_curve(p0, p1, p2, samples)
    samples = samples[nearest_stroke_indices.flatten()].view(img_size, img_size, k, n_samples, -1)

    dists = T.sqrt(T.sum((coords[:, :, None, None] - samples) ** 2, dim=-1) + 1e-8)  # HxWxkxS
    d_strokes = T.min(dists, dim=-1)[0]

    # stroke masks
    swidths = stroke_widths[nearest_stroke_indices.flatten()].view(img_size, img_size, k)
    m_strokes = T.max(T.sigmoid(temp * (swidths[..., None] - dists)), dim=-1)[0]  # HxWxk

    # render strokes
    I_strokes = m_strokes[..., None] * stroke_color[nearest_stroke_indices.flatten()].view(img_size, img_size, k, 3)

    # aggregate all strokes
    A = T.softmax(-300 * d_strokes, dim=-1)
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
