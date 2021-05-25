import torch as T
from torch.nn import functional as F
from torch_cluster import knn
import numpy as np
from torchvision.transforms import functional as TF

import utils


# ---------------------------------------------------------------------
# Brushstrokes
# ---------------------------------------------------------------------
def sample_quadratic_bezier_curve(s, c, e, num_points=20):
    """
    Samples points from the quadratic bezier curves defined by the control points.
    Number of points to sample is num.

    Args:
        s (tensor): Start point of each curve, shape [N, 2].
        c (tensor): Control point of each curve, shape [N, 2].
        e (tensor): End point of each curve, shape [N, 2].
        num_points (int): Number of points to sample on every curve.

    Return:
       (tensor): Coordinates of the points on the Bezier curves, shape [N, num_points, 2]
    """
    N, _ = s.shape
    t = T.linspace(0., 1., num_points).to(s.device)
    t = T.stack([t] * N, dim=0)
    s_x = s[..., 0:1]
    s_y = s[..., 1:2]
    e_x = e[..., 0:1]
    e_y = e[..., 1:2]
    c_x = c[..., 0:1]
    c_y = c[..., 1:2]
    x = c_x + (1. - t) ** 2 * (s_x - c_x) + t ** 2 * (e_x - c_x)
    y = c_y + (1. - t) ** 2 * (s_y - c_y) + t ** 2 * (e_y - c_y)
    return T.stack([x, y], dim=-1)


@T.jit.script
def stroke_renderer(curve_points: T.Tensor, locations: T.Tensor, colors: T.Tensor, widths: T.Tensor,
                    H: int, W: int, K: int, canvas_color: float):
    """
    Renders the given brushstroke parameters onto a canvas.
    See Alg. 1 in https://arxiv.org/pdf/2103.17185.pdf.

    Args:
        curve_points (tensor): Points specifying the curves that will be rendered on the canvas, shape [N, S, 2].
        locations (tensor): Location of each curve, shape [N, 2].
        colors (tensor): Color of each curve, shape [N, 3].
        widths (tensor): Width of each curve, shape [N, 1].
        H (int): Height of the canvas.
        W (int): Width of the canvas.
        K (int): Number of brushstrokes to consider for each pixel, see Sec. C.2 of the paper (Arxiv version).
        canvas_color (str): Background color of the canvas. Options: 'gray', 'white', 'black', 'noise'.
    Returns:
        (tensor): The rendered canvas, shape [H, W, 3].
    """
    colors = T.clamp(colors, 0., 1.)
    coord_x, coord_y = T.split(locations, [1, 1], dim=-1)
    coord_x = T.clamp(coord_x, 0, W)
    coord_y = T.clamp(coord_y, 0, H)
    locations = T.cat((coord_x, coord_y), dim=1)
    widths = T.exp(widths)

    device = curve_points.device
    N, S, _ = curve_points.shape

    # define coarse grid cell
    t_H = T.linspace(0., float(H), int(H // 5)).to(device)
    t_W = T.linspace(0., float(W), int(W // 5)).to(device)
    P_y, P_x = T.meshgrid(t_H, t_W)
    P = T.stack([P_x, P_y], dim=-1)  # [32, 32, 2]

    # Find nearest brushstrokes' indices for every coarse grid cell
    indices = knn(locations, P.view(-1, 2), k=K)[1]

    # Resize the KNN index tensor to full resolution
    indices = indices.view(len(t_H), len(t_W), -1)
    indices = indices.permute(2, 0, 1)
    indices = TF.resize(indices, size=(H, W), interpolation=TF.InterpolationMode.NEAREST)
    indices = indices.permute(1, 2, 0)

    # locations of points sampled from curves
    canvas_with_nearest_Bs = curve_points[indices.flatten()].view(H, W, K, S, 2)

    # colors of curves
    canvas_with_nearest_Bs_colors = colors[indices.flatten()].view(H, W, K, 3)

    # brush size
    canvas_with_nearest_Bs_bs = widths[indices.flatten()].view(H, W, K, 1)

    # Now create full-size canvas
    t_H = T.linspace(0., float(H), H).to(device)
    t_W = T.linspace(0., float(W), W).to(device)
    P_y, P_x = T.meshgrid(t_H, t_W)
    P_full = T.stack([P_x, P_y], dim=-1)  # [H, W, 2]

    # Compute distance from every pixel on canvas to each (among nearest ones) line segment between points from curves
    indices_a = T.tensor([i for i in range(S - 1)], dtype=T.long).to(device)
    canvas_with_nearest_Bs_a = canvas_with_nearest_Bs[:, :, :, indices_a, :]  # start points of each line segment
    indices_b = T.tensor([i for i in range(1, S)], dtype=T.long).to(device)
    canvas_with_nearest_Bs_b = canvas_with_nearest_Bs[:, :, :, indices_b, :]  # end points of each line segments
    canvas_with_nearest_Bs_b_a = canvas_with_nearest_Bs_b - canvas_with_nearest_Bs_a  # [H, W, N, S - 1, 2]
    P_full_canvas_with_nearest_Bs_a = P_full[:, :, None, None, :] - canvas_with_nearest_Bs_a  # [H, W, K, S - 1, 2]

    # find the projection of grid points on curves
    # first find the projections of a grid point on each line segment of a curve
    # numerator is the dot product between two vectors
    # the first vector is the line segments. the second vector is the sample points -> grid
    t = T.sum(canvas_with_nearest_Bs_b_a * P_full_canvas_with_nearest_Bs_a, dim=-1) / (
            T.sum(canvas_with_nearest_Bs_b_a ** 2, dim=-1) + 1e-8)

    # if t value is outside [0, 1], then the nearest point on the line does not lie on the segment, so clip values of t
    t = T.clamp(t, 0., 1.)

    # compute closest points on each line segment, which are the projections on each segment - [H, W, K, S - 1, 2]
    closest_points_on_each_line_segment = canvas_with_nearest_Bs_a + t[..., None] * canvas_with_nearest_Bs_b_a

    # compute the distance from every pixel to the closest point on each line segment - [H, W, K, S - 1]
    dist_to_closest_point_on_line_segment = T.sum(
        (P_full[..., None, None, :] - closest_points_on_each_line_segment) ** 2, dim=-1)

    # and distance to the nearest bezier curve.
    D_per_strokes = T.amin(dist_to_closest_point_on_line_segment, dim=-1)  # [H, W, K]
    D = T.amin(D_per_strokes, dim=-1)  # [H, W]

    # Finally render curves on a canvas to obtain image.
    I_NNs_B_ranking = F.softmax(100000. * (1.0 / (1e-8 + D_per_strokes)), dim=-1)  # [H, W, N]
    I_colors = T.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_colors, I_NNs_B_ranking)  # [H, W, 3]
    bs = T.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_bs, I_NNs_B_ranking)  # [H, W, 1]
    bs_mask = T.sigmoid(bs - D[..., None])  # AOE of each brush stroke
    canvas = T.ones_like(I_colors) * canvas_color
    I = I_colors * bs_mask + (1 - bs_mask) * canvas
    return I  # HxWx3


class BrushStrokeRenderer(T.nn.Module):
    def __init__(self, canvas_height, canvas_width, num_strokes=5000, samples_per_curve=10, strokes_per_pixel=20,
                 canvas_color='gray', length_scale=1.1, width_scale=.1, content_img=None):
        super().__init__()
        if canvas_color == 'gray':
            self.canvas_color = .5
        elif canvas_color == 'black':
            self.canvas_color = 0.
        elif canvas_color == 'noise':
            self.canvas_color = T.rand(canvas_height, canvas_width, 3) * 0.1
        else:
            self.canvas_color = 1.

        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.num_strokes = num_strokes
        self.samples_per_curve = samples_per_curve
        self.strokes_per_pixel = strokes_per_pixel
        self.length_scale = length_scale
        self.width_scale = width_scale

        # brush stroke init
        if content_img is not None:
            location, s, e, c, width, color = utils.initialize_brushstrokes(content_img, num_strokes,
                                                                            canvas_height, canvas_width,
                                                                            length_scale, width_scale)
        else:
            location, s, e, c, width, color = utils.initialize_brushstrokes(content_img, num_strokes,
                                                                            canvas_height, canvas_width,
                                                                            length_scale, width_scale, init='random')
        location = location[..., ::-1]
        s = s[..., ::-1]
        e = e[..., ::-1]
        c = c[..., ::-1]
        self.curve_s = T.nn.Parameter(T.from_numpy(np.array(s, 'float32')), requires_grad=True)
        self.curve_e = T.nn.Parameter(T.from_numpy(np.array(e, 'float32')), requires_grad=True)
        self.curve_c = T.nn.Parameter(T.from_numpy(np.array(c, 'float32')), requires_grad=True)
        self.color = T.nn.Parameter(T.from_numpy(color), requires_grad=True)
        self.location = T.nn.Parameter(T.from_numpy(np.array(location, 'float32')), requires_grad=True)
        self.width = T.nn.Parameter(T.from_numpy(np.log(np.minimum(width, 1e-3))), requires_grad=True)

    def forward(self):
        curve_points = sample_quadratic_bezier_curve(s=self.curve_s + self.location,
                                                     e=self.curve_e + self.location,
                                                     c=self.curve_c + self.location,
                                                     num_points=self.samples_per_curve)
        canvas = stroke_renderer(curve_points, self.location, self.color, self.width,
                                 self.canvas_height, self.canvas_width, self.strokes_per_pixel, self.canvas_color)
        return canvas
