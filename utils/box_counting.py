import torch
import numpy as np
from torch.nn.functional import max_pool2d

from sklearn.linear_model import LinearRegression

def count_boxes(raster: torch.tensor, box_size: int) -> int:
    pooled = max_pool2d(
        raster.float().unsqueeze(0).unsqueeze(0),
        kernel_size=box_size
    )
    return pooled.sum()

def get_fractal_dimention(raster:torch.tensor):
    """
    Считает фрактальную размерность клетками размера 
    10*k, k = 1, 5
    """
    model = LinearRegression()
    box_counts = []
    box_sizes = []
    for i in range(1, 10):
        box_counts.append(
            count_boxes(
                raster,
                i*10
            )
        )
        box_sizes.append(i*10)
    model.fit(
        np.log(box_sizes).reshape(-1, 1), 
        -np.log(box_counts)
    )
    return model.coef_[0]


def get_fractal_dimension_adaptive(raster, min_size=2, num_sizes=15):
    h, w = raster.shape
    max_size = min(h, w) // 2  # не больше половины изображения
    if max_size < min_size:
        return np.nan
    # Логарифмически распределённые размеры
    sizes = np.unique(np.logspace(np.log10(min_size), np.log10(max_size), num_sizes, dtype=int))
    box_counts = []
    valid_sizes = []
    for size in sizes:
        try:
            cnt = count_boxes(raster, size)
            if cnt > 0:
                box_counts.append(cnt)
                valid_sizes.append(size)
        except:
            # Игнорируем размеры, вызвавшие ошибку
            pass
    if len(valid_sizes) < 3:
        return np.nan
    # Регрессия: -ln(counts) ~ ln(size)  (коэффициент = D)
    X = np.log(valid_sizes).reshape(-1, 1)
    y = -np.log(box_counts)
    model = LinearRegression().fit(X, y)
    return model.coef_[0]
        
        