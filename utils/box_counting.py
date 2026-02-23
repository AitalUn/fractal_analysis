import torch
import numpy as np
from torch.nn.functional import max_pool2d
from sklearn.linear_model import LinearRegression
import itertools
from torch.functional import F


def count_boxes(raster: torch.tensor, box_size: int) -> int:
    raster = torch.nan_to_num(raster, nan=0.0)
    pooled = max_pool2d(
        raster.float().unsqueeze(0).unsqueeze(0),
        kernel_size=box_size
    )
    return pooled.sum()


def get_fractal_dimention(raster: torch.tensor):
    box_size = max(raster.shape)
    kernel_sizes = [int(box_size / x) for x in [4, 8, 16, 32]]
    model = LinearRegression()
    box_counts = [count_boxes(raster, ks) for ks in kernel_sizes]
    if (0 in box_counts) or any(np.isnan(x) for x in box_counts):
        return len(raster.shape) 
    model.fit(
        np.log(kernel_sizes).reshape(-1, 1),
        -np.log(box_counts)
    )
    return model.coef_[0]


def make_multifractal_analysis(raster: torch.Tensor, return_to_original=True):
    patch_size = 64
    
    # Сохраняем исходный размер
    original_size = (raster.shape[0], raster.shape[1])

    n = raster.shape[0] // patch_size 
    m = raster.shape[1] // patch_size 

    dx = raster.shape[0] % patch_size // 2
    dy = raster.shape[1] % patch_size // 2

    cropped = raster[
        dx: dx+n*patch_size, dy:dy+m*patch_size
    ]

    patches = cropped.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    
    multifractal_raster = np.zeros((n, m))
    for x_patch, y_patch in itertools.product(range(n), range(m)):
        multifractal_raster[x_patch, y_patch] = get_fractal_dimention(patches[x_patch, y_patch, :, :])  
    
    if return_to_original:
        # Конвертируем в torch tensor и добавляем размерности для интерполяции
        result_tensor = torch.from_numpy(multifractal_raster).float()
        
        # Добавляем batch и channel dimensions: (1, 1, n, m)
        result_tensor = result_tensor.unsqueeze(0).unsqueeze(0)
        
        # Интерполяция до исходного размера
        upscaled = F.interpolate(
            result_tensor, 
            size=original_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Убираем добавленные размерности
        upscaled = upscaled.squeeze(0).squeeze(0)
        
        return upscaled.numpy()
    
    return multifractal_raster