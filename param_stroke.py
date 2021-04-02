import torch as T
from imageio import imwrite
from torch.nn import functional as F


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


def stroke_renderer(stroke_params: T.Tensor, n_samples=10, temp=8., img_size=256, k=20):
    """

    :param stroke_params: Nx12, \in [0, 1]
    :param temp: float
    :param n_samples: int
    :return:
    """
    p0, p1, p2, stroke_widths, stroke_color = T.split(stroke_params, [2, 2, 2, 1, 3], dim=-1)
    p0, p1, p2 = T.tanh(p0) / 2. + .5, T.tanh(p1) / 2. + .5, T.tanh(p2) / 2. + .5  # need this later
    stroke_color = T.tanh(stroke_color) / 2. + .5  # need it later
    stroke_widths = T.exp(stroke_widths) / img_size  # no negative, need it later
    # stroke_widths = stroke_widths / img_size

    coords = T.linspace(0, 1, img_size).to(stroke_params.device)
    coords = T.stack(T.meshgrid((coords, coords)), dim=2)
    coords_coarse = F.interpolate(coords.permute(2, 0, 1)[None], (int(.1 * img_size), int(.1 * img_size)))[0].permute(1, 2, 0).contiguous()
    strokes_pos = T.stack((p0, p1, p2), dim=1)
    dists = T.min(T.sum((coords_coarse[:, :, None, None] - strokes_pos[None, None]) ** 2, dim=-1), dim=-1)[0]
    nearest_stroke_indices = T.topk(dists, k, largest=False, dim=-1)[1]
    nearest_stroke_indices = F.interpolate(nearest_stroke_indices.float().permute(2, 0, 1)[None], (img_size, img_size), mode='nearest')[0].permute(1, 2, 0).contiguous().long()

    # samples = T.rand(stroke_params.shape[0], n_samples).to(stroke_params.device)
    samples = T.linspace(0, 1, 10).to(stroke_params.device)[None]
    samples = bezier_curve(p0, p1, p2, samples)
    samples = samples[nearest_stroke_indices.flatten()].view(img_size, img_size, k, n_samples, 2)

    dists = T.sqrt(T.sum((coords[:, :, None, None] - samples) ** 2, dim=-1) + 1e-8)  # HxWxNxS
    d_strokes = T.min(dists, dim=-1)[0]

    swidths = stroke_widths[nearest_stroke_indices.flatten()].view(img_size, img_size, k)
    m_strokes = T.sigmoid(temp * (swidths - d_strokes))
    I_strokes = m_strokes[..., None] * stroke_color[nearest_stroke_indices.flatten()].view(img_size, img_size, k, 3)
    A = T.softmax(-temp * d_strokes, dim=-1)
    I = T.sum(I_strokes * A[..., None], dim=-2)
    return I


if __name__ == '__main__':
    strokes = T.tensor([
        [.5, .5, .6, .2, .7, .3, 10., 0., 0., 1.],
        [.1, .2, .102, .14, .21, .3, 20., 1., 0., 0.],
        [.8, .2, .6, .3, .9, .35, 5., .0, 1, 0.]
    ])
    img = stroke_renderer(strokes, temp=8, k=3)
    imwrite('strokes.jpg', img.detach().numpy())
