import torch as T
from imageio import imwrite
from torch.nn import functional as F
from torch_cluster import knn


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
    widths = T.relu(widths)

    device = curve_points.device
    N, S, _ = curve_points.shape
    # define coarse grid cell
    t_H = T.linspace(0., float(H), int(H // 5)).to(device)
    t_W = T.linspace(0., float(W), int(W // 5)).to(device)
    P_y, P_x = T.meshgrid(t_H, t_W)
    P = T.stack([P_x, P_y], dim=-1)  # [32, 32, 2]

    # Compute now distances from every brushtroke center to every coarse grid cell
    # D_to_all_B_centers = T.sum((P[..., None, :] - locations) ** 2, dim=-1)  # [H // C, W // C, N]
    # Find nearest brushstrokes' indices for every coarse grid cell
    idcs = knn(locations, P.view(-1, 2), k=K)[1]

    # [H // 10, W // 10, K, S, 2]
    canvas_with_nearest_Bs = curve_points[idcs.view(-1)].view(len(t_H), len(t_W), K, S, 2)

    # [H // 10, W // 10, K, 3]
    canvas_with_nearest_Bs_colors = colors[idcs.view(-1)].view(len(t_H), len(t_W), K, 3)

    # [H // 10, W // 10, K, 1]
    canvas_with_nearest_Bs_bs = widths[idcs.view(-1)].view(len(t_H), len(t_W), K, 1)

    # Resize those tensors to the full canvas size (not coarse grid)
    # First locations of points sampled from curves
    H_, W_, r1, r2, r3 = canvas_with_nearest_Bs.shape
    canvas_with_nearest_Bs = canvas_with_nearest_Bs.view(1, H_, W_, r1 * r2 * r3)  # [1, H // 10, W // 10, K * S * 2]
    canvas_with_nearest_Bs = F.interpolate(canvas_with_nearest_Bs.permute(0, 3, 1, 2), size=[H, W], mode='nearest')
    canvas_with_nearest_Bs = canvas_with_nearest_Bs.permute(0, 2, 3, 1).view(H, W, r1, r2, r3)  # [H, W, N, S, 2]

    # Now colors of curves
    H_, W_, r1, r2 = canvas_with_nearest_Bs_colors.shape
    canvas_with_nearest_Bs_colors = canvas_with_nearest_Bs_colors.view(1, H_, W_,
                                                                       r1 * r2)  # [1, H // 10, W // 10, K * 3]
    canvas_with_nearest_Bs_colors = F.interpolate(canvas_with_nearest_Bs_colors.permute(0, 3, 1, 2), size=(H, W),
                                                  mode='nearest')
    canvas_with_nearest_Bs_colors = canvas_with_nearest_Bs_colors.permute(0, 2, 3, 1).view(H, W, r1, r2)  # [H, W, K, 3]

    # And with the brush size
    H_, W_, r1, r2 = canvas_with_nearest_Bs_bs.shape
    canvas_with_nearest_Bs_bs = canvas_with_nearest_Bs_bs.view(1, H_, W_, r1 * r2)  # [1, H // 10, W // 10, K]
    canvas_with_nearest_Bs_bs = F.interpolate(canvas_with_nearest_Bs_bs.permute(0, 3, 1, 2), size=(H, W),
                                              mode='nearest')
    canvas_with_nearest_Bs_bs = canvas_with_nearest_Bs_bs.permute(0, 2, 3, 1).view(H, W, r1, r2)  # [H, W, K, 1]

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

    # compute t value for which each pixel is closest to each line that goes through each line segment (among nearest ones)
    t = T.sum(canvas_with_nearest_Bs_b_a * P_full_canvas_with_nearest_Bs_a, dim=-1) / (
                T.sum(canvas_with_nearest_Bs_b_a ** 2, dim=-1) + 1e-8)

    # if t value is outside [0, 1], then the nearest point on the line does not lie on the segment, so clip values of t
    t = T.clamp(t, 0., 1.)

    # compute closest points on each line segment - [H, W, K, S - 1, 2]
    closest_points_on_each_line_segment = canvas_with_nearest_Bs_a + t[..., None] * canvas_with_nearest_Bs_b_a

    # compute the distance from every pixel to the closest point on each line segment - [H, W, K, S - 1]
    dist_to_closest_point_on_line_segment = T.sum(
        (P_full[..., None, None, :] - closest_points_on_each_line_segment) ** 2, dim=-1)

    # and distance to the nearest bezier curve.
    D = T.amin(dist_to_closest_point_on_line_segment, dim=(-1, -2))  # [H, W]

    # Finally render curves on a canvas to obtain image.
    I_NNs_B_ranking = F.softmax(100000. * (1.0 / (1e-8 + T.amin(dist_to_closest_point_on_line_segment, dim=-1))),
                                dim=-1)  # [H, W, N]
    I_colors = T.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_colors, I_NNs_B_ranking)  # [H, W, 3]
    bs = T.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_bs, I_NNs_B_ranking)  # [H, W, 1]
    bs_mask = T.sigmoid(bs - D[..., None])
    canvas = T.ones_like(I_colors) * canvas_color
    # if canvas_color == 'gray':
    #     canvas *= .5
    # elif canvas_color == 'black':
    #     canvas *= 0.
    # elif canvas_color == 'noise':
    #     canvas = T.randn_like(I_colors) * 0.1

    I = I_colors * bs_mask + (1 - bs_mask) * canvas
    return I  # HxWx3


if __name__ == '__main__':
    import math

    strokes = T.tensor([
        [.3, .5, .1, .1, .1, .2, .1, .1, math.log(10.), 0., 0., 1.],
        [.1, .2, .01, .01, .0102, .014, .021, .03, math.log(20.), 1., 0., 0.],
        [.8, .3, -.01, -.01, .01, .01, -.01, .01, math.log(5.), .0, 1, 0.]
    ])
    img = stroke_renderer(None, strokes, temp=150., k=2)
    imwrite('strokes.jpg', img.detach().numpy())
