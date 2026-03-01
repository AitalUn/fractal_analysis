import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


def gaussian_kernel(kernel_size: int, sigma: float, device):
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def smooth_fractal_map(D, kernel_size=5, sigma=1.0):
    """
    D: (H, W) tensor
    """
    device = D.device
    
    kernel = gaussian_kernel(kernel_size, sigma, device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)
    
    D = D.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    
    D_smooth = F.conv2d(
        D,
        kernel,
        padding=kernel_size // 2
    )
    
    return D_smooth.squeeze()


def fractal_dimension_map_2d(img: torch.Tensor,
                             threshold: float = None,
                             eps_list=(9, 27, 81),
                             window_size=81,
                             eps_small=1e-8):

    assert img.dim() == 2
    device = img.device
    dtype = torch.float32

    x = img.to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    if threshold is not None:
        x = (x > threshold).float()

    H, W = img.shape
    counts = []

    for eps in eps_list:

        # 1️⃣ огрубление
        pooled = F.max_pool2d(x, kernel_size=eps, stride=eps)

        # 2️⃣ бинаризация занятых боксов
        occupied = (pooled > 0).float()

        # 3️⃣ считаем количество боксов в 81×81 окне
        k = window_size // eps  # размер окна в coarse-сетке

        # паддинг чтобы вернуть исходный размер
        pad = k // 2

        # свёртка единичным ядром
        kernel = torch.ones((1, 1, k, k), device=device)
        local_count = F.conv2d(occupied, kernel, padding=pad)

        # upscale обратно к размеру H×W
        up = F.interpolate(local_count,
                           size=(H, W),
                           mode='nearest')

        counts.append(up.squeeze(0).squeeze(0))

    # stack: (3, H, W)
    N_eps = torch.stack(counts, dim=0)

    # логарифмы
    log_N = torch.log(N_eps + eps_small)

    log_eps = torch.log(torch.tensor(eps_list,
                                     dtype=dtype,
                                     device=device)).view(-1, 1, 1)

    x_mean = log_eps.mean()
    var_x = ((log_eps - x_mean) ** 2).sum()

    y_mean = log_N.mean(dim=0, keepdim=True)

    cov = ((log_eps - x_mean) * (log_N - y_mean)).sum(dim=0)

    slope = cov / var_x

    D = -slope

    # Сглаживание для подавления квадратного эффетка
    D = smooth_fractal_map(D, kernel_size=5, sigma=1.2)
    return D
