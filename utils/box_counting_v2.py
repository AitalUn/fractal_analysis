import torch
import numpy as np
from torch.nn.functional import max_pool2d
from sklearn.linear_model import LinearRegression
import itertools
from torch.functional import F

def count_boxes(raster: torch.Tensor, box_size: int) -> int:
    """Подсчет количества боксов с дополнением нулями"""
    raster = torch.nan_to_num(raster, nan=0.0)
    
    h, w = raster.shape
    pad_h = (box_size - h % box_size) % box_size
    pad_w = (box_size - w % box_size) % box_size
    
    raster_padded = torch.nn.functional.pad(
        raster.float(), (0, pad_w, 0, pad_h), mode='constant', value=0
    )
    
    pooled = torch.nn.functional.max_pool2d(
        raster_padded.unsqueeze(0).unsqueeze(0),
        kernel_size=box_size
    )
    return int(pooled.sum().item())

def get_fractal_dimension(raster: torch.Tensor):
    """Вычисление фрактальной размерности методом подсчета боксов"""
    box_size = max(raster.shape)
    kernel_sizes = [int(box_size / x) for x in [4, 8, 16, 32]]
    
    # Убираем нулевые и слишком маленькие размеры
    kernel_sizes = [k for k in kernel_sizes if k > 1]
    
    if len(kernel_sizes) < 2:  # нужно минимум 2 точки для регрессии
        return len(raster.shape)
    
    box_counts = [count_boxes(raster, ks) for ks in kernel_sizes]
    
    # Проверка на валидность для логарифмирования
    valid_pairs = []
    for ks, bc in zip(kernel_sizes, box_counts):
        if bc > 0:  # только положительные значения
            valid_pairs.append((ks, bc))
    
    if len(valid_pairs) < 2:  # недостаточно точек
        return len(raster.shape)
    
    # Разделяем на x и y
    valid_kernels, valid_counts = zip(*valid_pairs)
    
    model = LinearRegression()
    model.fit(
        np.log(valid_kernels).reshape(-1, 1),
        -np.log(valid_counts)
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
        multifractal_raster[x_patch, y_patch] = get_fractal_dimension(patches[x_patch, y_patch, :, :])  
    
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