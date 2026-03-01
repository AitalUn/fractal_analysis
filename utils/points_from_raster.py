import rasterio
import fiona
import numpy as np
from pyproj import Transformer

def extract_raster_values_at_points(raster_path, shp_path, shp_encoding="cp1251"):

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_nodata = src.nodata

        with fiona.open(shp_path, encoding=shp_encoding) as shp:

            shp_crs = shp.crs

            # Если CRS разные → создаём трансформер
            if shp_crs and raster_crs and shp_crs != raster_crs:
                transformer = Transformer.from_crs(
                    shp_crs,
                    raster_crs,
                    always_xy=True
                )
                coords = [
                    transformer.transform(*feat["geometry"]["coordinates"])
                    for feat in shp
                ]
            else:
                coords = [
                    feat["geometry"]["coordinates"]
                    for feat in shp
                ]

        # Получаем значения растра
        values = np.array([v[0] for v in src.sample(coords)])

        # Убираем nodata
        if raster_nodata is not None:
            mask = values != raster_nodata
            coords = np.array(coords)[mask]
            values = values[mask]

    return coords, values