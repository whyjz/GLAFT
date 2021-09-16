import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import map_coordinates
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from matplotlib import cm

def off_ice_errors(vfile=None, vxfile=None, vyfile=None, off_ice_area=None, thres_sigma=3.0, plot=True, ax=None):
    """
    vfile: str, geotiff file path
    vxfile: str, geotiff file path
    vyfile: str, geotiff file path
    off_ice_area: str, off ice area (shapefile) file path
    ----
    returns:
    vx: 1-d np array, vx values from all pixels within the off-ice area.
    vy: 1-d np array, vy values from all pixels within the off-ice area.
    z: 1-d np array (float), Gaussian KDE values for all pixels within the off-ice area.
    thres_idx: 1-d np array (boolean), indices of pixels with a z value within a pre-defined confidence level. 
               when thres_sigma=3.0, confidence level = 99.7%.
    ==== or ====
    v: 1-d np array, v values from all pixels within the off-ice area.
    bins: plt.hist return with 100 bins and v^2 as input
    ----
    according to
    https://gis.stackexchange.com/questions/260304/extract-raster-values-within-shapefile-with-pygeoprocessing-or-gdal
    """  
    shapefile = gpd.read_file(off_ice_area)
    geoms = shapefile.geometry.values
    geoms = [mapping(geoms[i]) for i in range(len(geoms))]
    
    def clip(gtiff, geoms):
        with rasterio.open(gtiff) as src:
            out_image, out_transform = mask(src, geoms, crop=True, nodata=-9999.0)
        try:
            clipped_data = out_image.data[0]
        except NotImplementedError:
            clipped_data = out_image[0]
        return clipped_data

    if vxfile is not None and vyfile is not None:
        case = 1
        vx = clip(vxfile, geoms)
        vy = clip(vyfile, geoms)
        vx = vx[vx > -9998]  # remove NaN points
        vy = vy[vy > -9998]  # remove NaN points
    elif vfile is not None:
        case = 2
        v = clip(vfile, geoms)
        v = v[v > -9998]  # remove NaN points
    else:
        case = 0
        raise TypeError('Either vfile or vxfile+vyfile are required.')
        
    if case == 1:
        xy = np.vstack([vx, vy])
        z = gaussian_kde(xy)(xy)

        thres_multiplier = np.e ** (thres_sigma ** 2 / 2)   # normal dist., +- sigma number 
        thres = max(z) / thres_multiplier
        thres_idx = z >= thres
        idx = thres_idx    # alias

        if plot:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                
            viridis = cm.get_cmap('viridis', 12)
            pt_style = {'s': 6, 'edgecolor': None}
            
            ax.scatter(vx[idx], vy[idx], c=z[idx], **pt_style)
            ax.scatter(vx[~idx], vy[~idx], color=viridis(0), alpha=0.4, **pt_style)
            
        return vx, vy, z, thres_idx
    
    elif case == 2:
        
        if plot:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                
        bins = ax.hist(v ** 2, 100);
        return v, bins

    
def create_synthetic_offset(imgfile, mode='subpixel', block_size=500):
    """
    imgfile: str, geotiff file path
    mode: 'subpixel' or 'multipixel'
    block_size: int, increment block size
    ----
    returns:
    shift_arx: np.ndarray, offset field (x), in pixels
    shift_ary: np.ndarray, offset field (y), in pixels
    ----

    """  
    with rasterio.open(imgfile) as src:
        data_shape = (src.height, src.width)
    idxy, idxx = np.indices(data_shape)   
    # for Numpy array, first is row element (-> geotiff's y direction, height) 
    # and second is column element (-> geotiff's x direction, width)
    
    if mode == 'subpixel':
        shift_arx = idxx // block_size
        shift_arx = 0.1 * shift_arx + 0.1
        shift_ary = idxy // block_size
        shift_ary = -0.1 * shift_ary - 0.1
    elif mode == 'multipixel':
        shift_arx = 1 + idxx // block_size
        shift_ary = -1 - idxy // block_size
    else:
        raise ValueError('Mode is not defined.')
        
    return shift_arx, shift_ary


def apply_synthetic_offset(imgfile, shift_arx, shift_ary, spline_order=1):
    """
    imgfile: str, geotiff file path
    shift_arx: np.ndarray, offset field (x) from gftt.create_synthetic_offset
    shift_ary: np.ndarray, offset field (y) from gftt.create_synthetic_offset
    ----
    returns:

    ----

    """  
    with rasterio.open(imgfile) as src:
        data_shape = (src.height, src.width)
        data = src.read(1)
    idxy, idxx = np.indices(data_shape)
    shifted_y = idxy + shift_ary
    shifted_x = idxx + shift_arx
    shifted_yx = np.vstack((shifted_y.flatten(), shifted_x.flatten()))
    
    shifted_val = map_coordinates(data, shifted_yx, order=spline_order, mode='nearest')
    shifted_val = np.reshape(shifted_val, data_shape)
    
    return shifted_val
    
def syn_shift_errors(ref_vx=None, vx=None, ref_vy=None, vy=None, thres_sigma=3.0, plot=True, ax=None):
    if ref_vx is not None and vx is not None and ref_vy is not None and vy is not None:
        ref_vx = ref_vx.flatten()
        vx = vx.flatten()
        ref_vy = ref_vy.flatten()
        vy = vy.flatten()
        valid_idx = vx > -9998 # idx of non-NaN points
        # valid_idx = vy > -99 # idx of non-NaN points
        # print('something')
        vx = vx[valid_idx]
        vy = vy[valid_idx]
        ref_vx = ref_vx[valid_idx]
        ref_vy = ref_vy[valid_idx]
    else:
        raise TypeError('ref_vx, vx, ref_vy, and vy are all required.')
        
    diff_vx = vx - ref_vx
    diff_vy = vy - ref_vy
    # return diff_vx, diff_vy
    
    print('Start calculating KDE, this may take a while...')        
    xy = np.vstack([diff_vx, diff_vy])
    z = gaussian_kde(xy)(xy)
    print('KDE Done!')

    thres_multiplier = np.e ** (thres_sigma ** 2 / 2)   # normal dist., +- sigma number 
    thres = max(z) / thres_multiplier
    thres_idx = z >= thres
    idx = thres_idx    # alias

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        viridis = cm.get_cmap('viridis', 12)
        pt_style = {'s': 6, 'edgecolor': None}

        ax.scatter(diff_vx[idx], diff_vy[idx], c=z[idx], **pt_style)
        ax.scatter(diff_vx[~idx], diff_vy[~idx], color=viridis(0), alpha=0.4, **pt_style)

    return diff_vx, diff_vy, z, thres_idx
    

