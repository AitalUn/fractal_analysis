import numpy as np
from typing import Sequence, Tuple, Optional, Dict
from math import ceil
import warnings

def partition_sum(grid: np.ndarray, box_size: int) -> np.ndarray:
    """
    Разбить квадратную матрицу grid (H_w x W_w) на не перекрывающиеся блоки box_size и
    вернуть 1D array сумм по блокам (в порядке row-major).
    Если размеры не кратны box_size — допадаем нулями справа/сверху.
    """
    H, W = grid.shape
    Hp = int(ceil(H / box_size)) * box_size
    Wp = int(ceil(W / box_size)) * box_size
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h or pad_w:
        grid_p = np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0.0)
    else:
        grid_p = grid
    # reshape trick
    Gh = grid_p.reshape((Hp // box_size, box_size, Wp // box_size, box_size))
    # sum over small blocks
    block_sums = Gh.sum(axis=(1, 3)).ravel()  # length = (Hp/box_size)*(Wp/box_size)
    return block_sums

def linear_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Обычная линейная регрессия y = a*x + b, возвращает a. Если недостаточно точек — None."""
    if len(x) < 2:
        return None
    # Только конечные положительные y (для лог)
    if np.any(np.isfinite(y)) == False:
        return None
    # if all equal -> slope 0
    try:
        a, b = np.polyfit(x, y, 1)
        return float(a)
    except Exception:
        return None

def multifractal_indices_window(window: np.ndarray,
                                r_list: Sequence[int],
                                q_list: Sequence[float],
                                eps: float = 1e-12
                                ) -> Dict[str, np.ndarray]:
    """
    Вычисляет для одного окна (2D array) мультифрактальные показатели.
    Возвращает dict со всеми промежуточными результатами:
      - 'tau' : array(len(q))
      - 'Dq'  : array(len(q))   (Dq for q !=1; D1 special)
      - 'D0','D1','D2' scalars (if q_list contains 0,1,2)
      - 'alpha' (len(q)), 'f_alpha' (len(q))
      - 'delta_alpha', 'delta_f'
    Параметры:
      window: 2D numpy array (неотрицательные значения, например массы/суммы).
      r_list: список размеров боксов (в пикселях), например [2,4,8,16]
      q_list: список q (может включать 0 и 1)
    """
    window = np.asarray(window, dtype=float)
    # если все нули -> возвращаем NaN-пакет
    if window.sum() <= eps:
        n = len(q_list)
        nanarr = np.full(n, np.nan)
        return {
            'tau': nanarr.copy(),
            'Dq': nanarr.copy(),
            'alpha': nanarr.copy(),
            'f_alpha': nanarr.copy(),
            'D0': np.nan, 'D1': np.nan, 'D2': np.nan,
            'delta_alpha': np.nan, 'delta_f': np.nan,
        }

    r_list = np.asarray(r_list, dtype=int)
    q_list = np.asarray(q_list, dtype=float)

    # Для каждого r вычислим Pi: сначала суммы Ni (по боксам), затем нормировка
    chi = np.zeros((len(q_list), len(r_list)), dtype=float)  # chi[q_idx, r_idx]
    S_for_D1 = np.zeros(len(r_list), dtype=float)  # sum Pi log Pi (shannon)
    log_r = np.log(r_list.astype(float))

    for j, r in enumerate(r_list):
        Ni = partition_sum(window, r)  # суммы в блоках
        Nt = Ni.sum()
        if Nt <= eps:
            # пустой на этом масштабе
            chi[:, j] = np.nan
            S_for_D1[j] = np.nan
            continue
        Pi = Ni / Nt
        # численно: убрать нули для логов
        nonzero = Pi > 0
        # chi
        for i_q, q in enumerate(q_list):
            if np.isclose(q, 0.0):
                # chi_0 = number of non-empty boxes
                chi[i_q, j] = np.count_nonzero(nonzero)
            else:
                chi[i_q, j] = np.sum(Pi[nonzero] ** q)
        # S (для q->1)
        S_for_D1[j] = np.sum(Pi[nonzero] * np.log(Pi[nonzero]))

    # Теперь для каждого q получаем tau(q) как slope of log chi vs log r
    tau = np.full(len(q_list), np.nan)
    for i_q, q in enumerate(q_list):
        y = chi[i_q, :]
        # valid indices where chi>0 and finite
        valid = np.isfinite(y) & (y > 0)
        if valid.sum() >= 2:
            log_y = np.log(y[valid])
            lr = linear_slope(log_r[valid], log_y)
            tau[i_q] = lr
        else:
            tau[i_q] = np.nan

    # Dq
    Dq = np.full_like(tau, np.nan)
    for i_q, q in enumerate(q_list):
        if np.isclose(q, 1.0):
            # D1 via S_for_D1: fit S(r) vs log r, slope = D1 (see Sun et al.)
            valid = np.isfinite(S_for_D1)
            if valid.sum() >= 2:
                slope_S = linear_slope(log_r[valid], S_for_D1[valid])
                Dq[i_q] = slope_S
            else:
                Dq[i_q] = np.nan
        else:
            if np.isfinite(tau[i_q]):
                Dq[i_q] = tau[i_q] / (q - 1.0)
            else:
                Dq[i_q] = np.nan

    # alpha and f(alpha) via Legendre transform: alpha = d tau / d q, f = q*alpha - tau
    # numerical derivative
    # ensure monotonic q_list for gradient; use np.gradient
    dq = np.gradient(q_list.astype(float))
    dtau_dq = np.gradient(tau, q_list)  # np handles uneven spacing
    alpha = dtau_dq
    f_alpha = q_list * alpha - tau

    # scalar summaries
    # find indices for q==0,q==1,q==2 if present
    def get_D_for(qval):
        idx = np.where(np.isclose(q_list, qval))[0]
        return (Dq[idx[0]] if idx.size else np.nan)

    D0 = get_D_for(0.0)
    D1 = get_D_for(1.0)
    D2 = get_D_for(2.0)

    # delta alpha and delta f
    if np.all(np.isfinite(alpha)):
        delta_alpha = np.nanmax(alpha) - np.nanmin(alpha)
    else:
        delta_alpha = np.nan
    if np.all(np.isfinite(f_alpha)):
        # they sometimes define delta_f = f(alpha_min)-f(alpha_max)
        delta_f = np.nanmax(f_alpha) - np.nanmin(f_alpha)
    else:
        delta_f = np.nan

    return {
        'tau': tau,
        'Dq': Dq,
        'alpha': alpha,
        'f_alpha': f_alpha,
        'D0': float(D0) if np.isfinite(D0) else np.nan,
        'D1': float(D1) if np.isfinite(D1) else np.nan,
        'D2': float(D2) if np.isfinite(D2) else np.nan,
        'delta_alpha': float(delta_alpha) if np.isfinite(delta_alpha) else np.nan,
        'delta_f': float(delta_f) if np.isfinite(delta_f) else np.nan,
    }

# def compute_spatial_multifractal_maps(data: np.ndarray,
#                                      window_size: int,
#                                      r_list: Sequence[int],
#                                      q_list: Sequence[float],
#                                      stride: Optional[int] = None,
#                                      mode: str = 'sliding'
#                                      ) -> Dict[str, np.ndarray]:
#     """
#     Применяет multifractal_indices_window к каждому окну по сетке и формирует карты.
#     Параметры:
#       - data: 2D numpy array (H x W), значения >=0 (если есть отрицательные — отсечь или взять abs)
#       - window_size: размер окна в пикселях (целое)
#       - r_list: list of box sizes (в пикселях) используемых для multifractal расчёта (например [1,2,4,8,16])
#       - q_list: список q (например np.linspace(-5,5,21))
#       - stride: шаг сдвига окна; если None -> stride = window_size (неперекрывающие блоки)
#       - mode: 'sliding' или 'block' ; 'block' эквивалент stride=window_size
#     Возвращает dict карт:
#       - 'D0','D1','D2','delta_alpha','delta_f'
#       - а также 'count' — количество непустых окон, 'centers' — координаты центров окон (list)
#     Размер выходных карт зависит от режима:
#       - если stride != window_size -> создаются карты с центрами для каждого окна (Hout x Wout)
#       - если block -> карта размера (H//window_size, W//window_size)
#     """
#     H, W = data.shape
#     if stride is None or mode == 'block':
#         stride = window_size

#     # compute number of windows in x,y
#     nx = 1 + (W - window_size) // stride if W >= window_size else 0
#     ny = 1 + (H - window_size) // stride if H >= window_size else 0
#     if nx <= 0 or ny <= 0:
#         raise ValueError("window_size too large for the input data")

#     # prepare output arrays
#     D0_map = np.full((ny, nx), np.nan, dtype=float)
#     D1_map = np.full((ny, nx), np.nan, dtype=float)
#     D2_map = np.full((ny, nx), np.nan, dtype=float)
#     dalpha_map = np.full((ny, nx), np.nan, dtype=float)
#     df_map = np.full((ny, nx), np.nan, dtype=float)

#     centers = []
#     for iy in range(ny):
#         y0 = iy * stride
#         for ix in range(nx):
#             x0 = ix * stride
#             window = data[y0:y0+window_size, x0:x0+window_size]
#             res = multifractal_indices_window(window, r_list, q_list)
#             D0_map[iy, ix] = res['D0']
#             D1_map[iy, ix] = res['D1']
#             D2_map[iy, ix] = res['D2']
#             dalpha_map[iy, ix] = res['delta_alpha']
#             df_map[iy, ix] = res['delta_f']
#             centers.append((y0 + window_size//2, x0 + window_size//2))

#     return {
#         'D0': D0_map,
#         'D1': D1_map,
#         'D2': D2_map,
#         'delta_alpha': dalpha_map,
#         'delta_f': df_map,
#         'centers': centers,
#         'nx': nx, 'ny': ny
#     }

# ------------------------------
# Пример использования:
# ------------------------------

from scipy.interpolate import RBFInterpolator

def compute_spatial_multifractal_maps(
        data: np.ndarray,
        window_size: int,
        r_list: Sequence[int],
        q_list: Sequence[float],
        stride: Optional[int] = None,
        mode: str = 'sliding',
        interpolate_to_full: bool = True,
        rbf_kernel: str = "thin_plate_spline"
    ) -> Dict[str, np.ndarray]:

    H, W = data.shape

    if stride is None or mode == 'block':
        stride = window_size

    nx = 1 + (W - window_size) // stride if W >= window_size else 0
    ny = 1 + (H - window_size) // stride if H >= window_size else 0
    if nx <= 0 or ny <= 0:
        raise ValueError("window_size too large for the input data")

    D0_map = np.full((ny, nx), np.nan, dtype=float)
    D1_map = np.full((ny, nx), np.nan, dtype=float)
    D2_map = np.full((ny, nx), np.nan, dtype=float)
    dalpha_map = np.full((ny, nx), np.nan, dtype=float)
    df_map = np.full((ny, nx), np.nan, dtype=float)

    centers = []

    for iy in range(ny):
        y0 = iy * stride
        for ix in range(nx):
            x0 = ix * stride
            window = data[y0:y0+window_size, x0:x0+window_size]
            res = multifractal_indices_window(window, r_list, q_list)

            D0_map[iy, ix] = res['D0']
            D1_map[iy, ix] = res['D1']
            D2_map[iy, ix] = res['D2']
            dalpha_map[iy, ix] = res['delta_alpha']
            df_map[iy, ix] = res['delta_f']

            centers.append((y0 + window_size//2, x0 + window_size//2))

    result = {
        'D0_coarse': D0_map,
        'D1_coarse': D1_map,
        'D2_coarse': D2_map,
        'delta_alpha_coarse': dalpha_map,
        'delta_f_coarse': df_map,
        'centers': centers
    }

    # ===============================
    # Интерполяция к исходному размеру
    # ===============================
    if interpolate_to_full:

        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        grid_points = np.column_stack([yy.ravel(), xx.ravel()])
        points = np.array(centers)

        def interpolate_field(field):

            values = field.flatten()
            valid = np.isfinite(values)

            if valid.sum() < 4:
                return np.full((H, W), np.nan)

            rbf = RBFInterpolator(
                points[valid],
                values[valid],
                kernel=rbf_kernel
            )

            interpolated = rbf(grid_points).reshape(H, W)
            return interpolated

        result['D0'] = interpolate_field(D0_map)
        result['D1'] = interpolate_field(D1_map)
        result['D2'] = interpolate_field(D2_map)
        result['delta_alpha'] = interpolate_field(dalpha_map)
        result['delta_f'] = interpolate_field(df_map)

    return result