import rasterio
import numpy as np
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
import pwlf
import matplotlib.pyplot as plt
import warnings

def rgb_to_hex(rgb_str):
    """Convert 'R,G,B,A' to '#RRGGBB' (alpha ignored)."""
    r, g, b, *_ = map(int, rgb_str.split(","))
    return f"#{r:02x}{g:02x}{b:02x}"

def lighten_color(rgb_str, factor=0.5):
    """
    Return a lighter version of the color.
    rgb_str: 'R,G,B,A' format.
    factor: 0 = original, 1 = white.
    """
    r, g, b, a = map(int, rgb_str.split(","))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"{r},{g},{b},{a}"

def get_ca_data(data, sampling_step=10, n_segments=3, low_quantile=0.05, high_quantile=0.95, n_points=30):
    """
    Compute C‑A fractal thresholds and return data for plotting.
    
    Returns
    -------
    dict with keys:
        'thresholds' : list of float (internal breakpoints in original scale)
        'logx' : array
        'logy' : array
        'model' : pwlf.PiecewiseLinFit object
        'x_vals' : array (original thresholds)
        'y_vals' : array (original areas)
        'breaks_log' : array (breakpoints in log scale)
    """
    # Subsample
    data_sampled = data[::sampling_step]
    if len(data_sampled) < 100:
        data_sampled = data  # fallback if too small

    # Define threshold grid
    vmin = np.quantile(data_sampled, low_quantile)
    vmax = np.quantile(data_sampled, high_quantile)
    x_vals = np.linspace(vmin, vmax, n_points)
    y_vals = np.array([np.sum(data_sampled < t) for t in x_vals])

    # Log transform (add epsilon to avoid log(0))
    eps = 1e-12
    logx = np.log(x_vals + eps)
    logy = np.log(y_vals + eps)

    # Fit piecewise linear model
    model = pwlf.PiecewiseLinFit(logx, logy)
    breaks_log = model.fit(n_segments)

    # Convert back to original scale, return internal breakpoints
    breaks_orig = np.exp(breaks_log)
    internal_breaks = breaks_orig[1:-1].tolist()

    return {
        'thresholds': sorted(internal_breaks),
        'logx': logx,
        'logy': logy,
        'model': model,
        'x_vals': x_vals,
        'y_vals': y_vals,
        'breaks_log': breaks_log
    }

def save_ca_plot(ca_data, output_png, raster_name=""):
    """Save log-log plot with piecewise linear fit."""
    logx = ca_data['logx']
    logy = ca_data['logy']
    model = ca_data['model']
    breaks_log = ca_data['breaks_log']

    plt.figure(figsize=(8, 6))
    plt.scatter(logx, logy, s=20, alpha=0.6, label='Data')
    
    x_smooth = np.linspace(logx.min(), logx.max(), 200)
    y_smooth = model.predict(x_smooth)
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Piecewise linear fit')
    
    for b in breaks_log[1:-1]:
        plt.axvline(b, color='gray', linestyle='--', alpha=0.7)
        plt.text(b, plt.ylim()[1]*0.95, f'{b:.2f}', rotation=90, verticalalignment='top')
    
    plt.xlabel('log(threshold)')
    plt.ylabel('log(area)')
    plt.title(f'C‑A fractal plot {raster_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

def generate_qml_ca(
    raster_path,
    qml_path,
    base_color,
    sampling_step=10,
    n_segments=3,
    low_quantile=0.05,
    high_quantile=0.95,
    n_points=30,
    fallback_percentiles=(70, 80, 95),
    save_plot=True
):
    """
    Generate QML style using C‑A fractal thresholds and optionally save C‑A plot.
    """
    raster_path = Path(raster_path)
    qml_path = Path(qml_path)

    # Read raster data
    with rasterio.open(raster_path) as src:
        band = src.read(1).astype("float64")
        nodata = src.nodata
        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)
        data = band[np.isfinite(band)]
        if data.size == 0:
            raise ValueError(f"No valid data in {raster_path}")
        vmin_data = float(np.nanmin(band))
        vmax_data = float(np.nanmax(band))

    # Compute C‑A thresholds
    ca_data = None
    try:
        ca_data = get_ca_data(
            data,
            sampling_step=sampling_step,
            n_segments=n_segments,
            low_quantile=low_quantile,
            high_quantile=high_quantile,
            n_points=n_points
        )
        thresholds = ca_data['thresholds']
        print(f"   C‑A thresholds: {[f'{t:.4f}' for t in thresholds]}")
    except Exception as e:
        warnings.warn(f"C‑A failed: {e}. Falling back to percentiles.")
        thresholds = []

    # Determine low, mid, high thresholds
    if len(thresholds) >= 3:
        low_th = thresholds[0]
        mid_th = thresholds[1]
        high_th = thresholds[2]
    elif len(thresholds) == 2:
        low_th = thresholds[0]
        mid_th = thresholds[1]
        high_th = np.percentile(data, fallback_percentiles[2])
    elif len(thresholds) == 1:
        low_th = thresholds[0]
        mid_th = np.percentile(data, fallback_percentiles[1])
        high_th = np.percentile(data, fallback_percentiles[2])
    else:
        low_th = np.percentile(data, fallback_percentiles[0])
        mid_th = np.percentile(data, fallback_percentiles[1])
        high_th = np.percentile(data, fallback_percentiles[2])

    low_th, mid_th, high_th = sorted([low_th, mid_th, high_th])

    # Save C‑A plot if requested and data available
    if save_plot and ca_data is not None:
        plot_path = qml_path.with_suffix('.ca.png')
        save_ca_plot(ca_data, plot_path, raster_name=raster_path.stem)
        print(f"   C‑A plot saved: {plot_path}")

    # Light colour at mid, full colour at high
    light_color = lighten_color(base_color, factor=0.7)
    dark_color = base_color

    # Build QML (same structure)
    root = Element("qgis", version="3.40.8-Bratislava", styleCategories="Symbology")
    pipe = SubElement(root, "pipe")

    renderer = SubElement(
        pipe,
        "rasterrenderer",
        type="singlebandpseudocolor",
        band="1",
        classificationMin=str(vmin_data),
        classificationMax=str(vmax_data),
        opacity="1",
        alphaBand="-1"
    )

    minmax = SubElement(renderer, "minMaxOrigin")
    SubElement(minmax, "limits").text = "MinMax"
    SubElement(minmax, "extent").text = "WholeRaster"
    SubElement(minmax, "statAccuracy").text = "Exact"

    shader = SubElement(renderer, "rastershader")
    crs = SubElement(
        shader,
        "colorrampshader",
        colorRampType="INTERPOLATED",
        minimumValue=str(vmin_data),
        maximumValue=str(vmax_data),
        clip="0",
        labelPrecision="4"
    )

    # Colour ramp definition (optional)
    ramp = SubElement(crs, "colorramp", type="gradient", name="[source]")
    opt = SubElement(ramp, "Option", type="Map")
    SubElement(opt, "Option", name="color1", type="QString", value=rgb_to_hex(light_color))
    SubElement(opt, "Option", name="color2", type="QString", value=rgb_to_hex(dark_color))
    SubElement(opt, "Option", name="discrete", type="QString", value="0")
    SubElement(opt, "Option", name="rampType", type="QString", value="gradient")
    SubElement(opt, "Option", name="spec", type="QString", value="rgb")

    # Items
    SubElement(
        crs,
        "item",
        value=f"{low_th:.6f}",
        color=rgb_to_hex(dark_color),
        label=f"< {low_th:.4f}",
        alpha="0"
    )
    SubElement(
        crs,
        "item",
        value=f"{mid_th:.6f}",
        color=rgb_to_hex(light_color),
        label=f"{low_th:.4f} – {mid_th:.4f}",
        alpha="0"
    )
    SubElement(
        crs,
        "item",
        value=f"{high_th:.6f}",
        color=rgb_to_hex(dark_color),
        label=f"{mid_th:.4f} – {high_th:.4f}",
        alpha="255"
    )

    # Save QML
    xml_str = minidom.parseString(tostring(root)).toprettyxml(indent="  ")
    with open(qml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    print(f"✅ QML generated for {raster_path.name} -> {qml_path.name}")
    print(f"   Thresholds: low={low_th:.4f}, mid={mid_th:.4f}, high={high_th:.4f}")

def generate_qmls_for_folder_ca(
    folder_path,
    colors_list=None,
    sampling_step=10,
    n_segments=3,
    low_quantile=0.05,
    high_quantile=0.95,
    n_points=30,
    fallback_percentiles=(70, 80, 95),
    pattern="*.tif",
    save_plot=True
):
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    if colors_list is None:
        colors_list = [
            "255,0,0,255",     # red
            "0,255,0,255",     # green
            "0,0,255,255",     # blue
            "255,165,0,255",   # orange
            "128,0,128,255",   # purple
            "255,192,203,255", # pink
            "0,255,255,255",   # cyan
            "255,255,0,255",   # yellow
            "165,42,42,255",   # brown
            "0,128,128,255",   # teal
        ]

    raster_files = sorted(folder.glob(pattern))
    if not raster_files:
        print(f"No files matching {pattern} found in {folder}")
        return

    for idx, raster_path in enumerate(raster_files):
        base_color = colors_list[idx % len(colors_list)]
        qml_path = raster_path.with_suffix(".qml")
        try:
            generate_qml_ca(
                raster_path,
                qml_path,
                base_color,
                sampling_step=sampling_step,
                n_segments=n_segments,
                low_quantile=low_quantile,
                high_quantile=high_quantile,
                n_points=n_points,
                fallback_percentiles=fallback_percentiles,
                save_plot=save_plot
            )
        except Exception as e:
            print(f"❌ Failed for {raster_path.name}: {e}")

if __name__ == "__main__":
    folder = r"D:\ml_datasets\Chukotka\Landsat_raw\SAM_indicies_landsat89_coged"
    generate_qmls_for_folder_ca(
        folder,
        sampling_step=10,
        n_segments=3,
        fallback_percentiles=(70, 80, 95),
        save_plot=True
    )