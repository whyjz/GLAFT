import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask

def off_ice_errors(ft_results=None, off_ice_area=None):
    """
    ft_results: str, geotiff file path
    off_ice_area: str, off ice area (shapefile) file path
    ----
    return all pixel values within a given polygon shapefile.
    according to
    https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal
    """  
    shapefile = gpd.read_file(off_ice_area)
    geoms = shapefile.geometry.values
    geoms = [mapping(geoms[i]) for i in range(len(geoms))]
    with rasterio.open(ft_results) as src:
        out_image, out_transform = mask(src, geoms, crop=True, nodata=-9999.0)
    try:
        clipped_data = out_image.data[0]
    except NotImplementedError:
        clipped_data = out_image[0]
    return clipped_data

