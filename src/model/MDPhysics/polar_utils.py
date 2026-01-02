import torch
import numpy as np
from torch import inf


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    # Ensure steps is on the same device as start
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


def get_sample_params_from_subdiv(
    subdiv,
    n_radius,
    n_azimuth,
    distortion_model,
    img_size,
    D,
    radius_buffer=0,
    azimuth_buffer=0,
):
    """Generate the required parameters to sample every patch based on the subdivison"""
    max_radius = min(img_size) / 2
    width = img_size[1]

    device = D.device

    if distortion_model == "spherical":
        fov = D[2][0]
        f = D[1]
        xi = D[0]
        D_min, theta_max = get_inverse_dist_spherical(
            subdiv[0], xi, fov, f, device=device
        )
    elif distortion_model == "polynomial":
        D_min, theta_max = get_inverse_distortion(subdiv[0], D, max_radius)

    D_s = torch.diff(D_min, axis=0)
    alpha = 2 * torch.tensor(np.pi, device=device) / subdiv[1]

    D_min = (
        D_min[:-1]
        .reshape(1, subdiv[0], D.shape[1])
        .repeat_interleave(subdiv[1], 0)
        .reshape(subdiv[0] * subdiv[1], D.shape[1])
    )
    D_s = (
        D_s.reshape(1, subdiv[0], D.shape[1])
        .repeat_interleave(subdiv[1], 0)
        .reshape(subdiv[0] * subdiv[1], D.shape[1])
    )
    phi_start = 0
    phi_end = 2 * torch.tensor(np.pi, device=device)
    phi_step = alpha
    phi_list = torch.arange(phi_start, phi_end, phi_step, device=device)
    p = phi_list.reshape(1, subdiv[1]).repeat_interleave(subdiv[0], 0)
    phi = p.transpose(1, 0).flatten()
    alpha = alpha.repeat_interleave(subdiv[0] * subdiv[1])

    if distortion_model == "spherical":
        params = {
            "alpha": alpha,
            "phi": phi,
            "dmin": D_min,
            "ds": D_s,
            "n_azimuth": n_azimuth,
            "n_radius": n_radius,
            "img_size": img_size,
            "radius_buffer": radius_buffer,
            "azimuth_buffer": azimuth_buffer,
            "subdiv": subdiv,
            "fov": fov,
            "xi": xi,
            "focal": f,
            "distort": distortion_model,
        }
    elif distortion_model == "polynomial":
        params = {
            "alpha": alpha,
            "phi": phi,
            "dmin": D_min,
            "ds": D_s,
            "n_azimuth": n_azimuth,
            "n_radius": n_radius,
            "img_size": img_size,
            "radius_buffer": radius_buffer,
            "azimuth_buffer": azimuth_buffer,
            "subdiv": subdiv,
            "distort": distortion_model,
        }

    return (
        params,
        D_s.reshape(subdiv[1], subdiv[0], D.shape[1]).permute(2, 1, 0),
        theta_max,
    )


def get_sample_locations(
    alpha,
    phi,
    dmin,
    ds,
    n_azimuth,
    n_radius,
    img_size,
    subdiv,
    distort,
    fov=0,
    focal=0,
    xi=0,
    radius_buffer=0,
    azimuth_buffer=0,
    img_B=1,
):
    """Get the sample locations in a given radius and azimuth range"""

    new_f = focal
    rad = (
        lambda x: new_f * torch.sin(torch.arctan(x)) / (xi + torch.cos(torch.arctan(x)))
    )
    inverse_rad = lambda r: torch.tan(
        torch.arcsin(xi * r / (new_f) / torch.sqrt(1 + (r / (new_f)) * (r / (new_f))))
        + torch.arctan(r / (new_f))
    )

    center = [img_size[0] / 2, img_size[1] / 2]
    if img_size[0] % 2 == 0:
        center[0] -= 0.5
    if img_size[1] % 2 == 0:
        center[1] -= 0.5

    r_end = dmin + ds
    r_start = dmin
    alpha_start = phi
    B = img_B
    alpha_end = alpha + phi

    radius = r_start.unsqueeze(0)
    radius = torch.transpose(radius, 0, 1)
    radius = radius.reshape(radius.shape[0] * radius.shape[1], B)

    azimuth = alpha_start.unsqueeze(0)
    azimuth = torch.transpose(azimuth, 0, 1)
    azimuth = azimuth.flatten()
    azimuth = azimuth.reshape(azimuth.shape[0], 1).repeat_interleave(B, 1)

    azimuth = azimuth.reshape(1, azimuth.shape[0], B).repeat_interleave(n_radius, 0)
    radius = radius.reshape(radius.shape[0], 1, B).repeat_interleave(n_azimuth, 1)

    radius_mesh = radius.reshape(subdiv[0] * subdiv[1], n_radius, n_azimuth, B)
    azimuth_mesh = azimuth.reshape(
        n_radius, subdiv[0] * subdiv[1], n_azimuth, B
    ).transpose(0, 1)

    azimuth_mesh_cos = torch.cos(azimuth_mesh)
    azimuth_mesh_sine = torch.sin(azimuth_mesh)
    x = radius_mesh * azimuth_mesh_cos
    y = radius_mesh * azimuth_mesh_sine

    return x.reshape(subdiv[0] * subdiv[1], n_radius * n_azimuth, B).transpose(
        1, 2
    ).transpose(0, 1), y.reshape(
        subdiv[0] * subdiv[1], n_radius * n_azimuth, B
    ).transpose(
        1, 2
    ).transpose(
        0, 1
    )


def get_inverse_distortion(num_points, D, max_radius):
    device = D.device
    dist_func = lambda x: x.reshape(1, x.shape[0]).repeat_interleave(
        D.shape[1], 0
    ).flatten() * (
        1
        + torch.outer(D[0], x**2).flatten()
        + torch.outer(D[1], x**4).flatten()
        + torch.outer(D[2], x**6).flatten()
        + torch.outer(D[3], x**8).flatten()
    )

    theta_max = dist_func(torch.tensor([1], device=device))

    theta = linspace(torch.tensor([0], device=device), theta_max, num_points + 1).to(
        device
    )

    test_radius = torch.linspace(0, 1, 50, device=device)
    test_theta = dist_func(test_radius).reshape(D.shape[1], 50).transpose(1, 0)

    radius_list = torch.zeros(num_points * D.shape[1], device=device).reshape(
        num_points, D.shape[1]
    )

    for i in range(D.shape[1]):
        for j in range(num_points):
            lower_idx = test_theta[:, i][test_theta[:, i] <= theta[:, i][j]].argmax()
            upper_idx = lower_idx + 1

            # Clamp upper_idx to avoid out of bounds (edge case)
            if upper_idx >= len(test_radius):
                upper_idx = len(test_radius) - 1

            x_0, x_1 = test_radius[lower_idx], test_radius[upper_idx]
            y_0, y_1 = test_theta[:, i][lower_idx], test_theta[:, i][upper_idx]

            # Avoid division by zero
            denom = y_1 - y_0
            # Use a small epsilon to prevent division by zero
            denom = torch.where(denom == 0, torch.tensor(1e-8, device=device), denom)
            radius_list[:, i][j] = x_0 + (theta[:, i][j] - y_0) * (x_1 - x_0) / denom

    max_rad = torch.tensor([1] * D.shape[1], device=device).reshape(1, D.shape[1])
    return torch.cat((radius_list, max_rad), axis=0) * max_radius, theta_max


def get_inverse_dist_spherical(num_points, xi, fov, new_f, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rad = (
        lambda x: new_f * torch.sin(torch.arctan(x)) / (xi + torch.cos(torch.arctan(x)))
    )
    inverse_rad = lambda r: torch.tan(
        torch.arcsin(xi * r / (new_f) * (1 + (r / (new_f)) * (r / (new_f))))
        + torch.arctan(r / (new_f))
    )

    min_val = inverse_rad(2.0)  # Variable name 'min' conflicts with built-in
    theta_d_max = torch.tan(fov / 2).to(device)
    theta_d = linspace(
        torch.tensor([0], device=device), theta_d_max, num_points + 1
    ).to(device)

    r_list = rad(theta_d)
    return r_list, theta_d_max


def get_sample_locations_reverse(H, W, n_azimuth, n_radius, subdiv, D, img_B=1):
    B = img_B
    x, theta = get_inverse_distortion(subdiv[0], D, float(H))
    x = (
        x[:-1]
        .reshape(1, subdiv[0], D.shape[1])
        .repeat_interleave(subdiv[1], 0)
        .reshape(subdiv[0] * subdiv[1], D.shape[1])
    )

    y, theta = get_inverse_distortion(subdiv[0], D, float(W))
    y = (
        y[:-1]
        .reshape(1, subdiv[0], D.shape[1])
        .repeat_interleave(subdiv[1], 0)
        .reshape(subdiv[0] * subdiv[1], D.shape[1])
    )

    x = torch.transpose(x, 0, 1)
    x = x.reshape(x.shape[0] * x.shape[1], B)

    y = torch.transpose(y, 0, 1)
    y = y.reshape(x.shape[0] * x.shape[1], B)

    y = y.reshape(1, y.shape[0], B).repeat_interleave(n_radius, 0)
    x = x.reshape(x.shape[0], 1, B).repeat_interleave(n_azimuth, 1)

    x = x.reshape(H * W, n_radius, n_azimuth, B)
    y = y.reshape(n_radius, H * W, n_azimuth, B).transpose(0, 1)

    radius = torch.sqrt(x**2 + y**2)
    azimuth = torch.atan2(y, x)

    return radius.reshape(H * W, n_radius * n_azimuth, B).transpose(1, 2).transpose(
        0, 1
    ), azimuth.reshape(H * W, n_radius * n_azimuth, B).transpose(1, 2).transpose(0, 1)
