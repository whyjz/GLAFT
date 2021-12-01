import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import map_coordinates, sobel
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask
from rasterio import features
import matplotlib.pyplot as plt
from matplotlib import cm


def off_ice_errors(vfile=None, vxfile=None, vyfile=None, wfile=None, off_ice_area=None, thres_sigma=3.0, plot=True, ax=None, max_n=10000):
    """
    vfile: str, geotiff file path
    vxfile: str, geotiff file path
    vyfile: str, geotiff file path
    wfile: str, goetiff file path (as weight)
    off_ice_area: str, off ice area (shapefile) file path
    max_n: maximum samples to calculate Gaussian KDE
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
    
    vx_full = None
    vy_full = None
    w_full = None

    if vxfile is not None and vyfile is not None:
        case = 1
        vx_full = clip(vxfile, geoms)
        vy_full = clip(vyfile, geoms)
        nonNaN_pts_idx = np.logical_and(vx_full > -9998, vy_full > -9998)
        vx_full = vx_full[nonNaN_pts_idx]  # remove NaN points
        vy_full = vy_full[nonNaN_pts_idx]  # remove NaN points
        if wfile is not None:
            w_full = clip(wfile, geoms)
            w_full = w_full[nonNaN_pts_idx]  # remove NaN points
        # return vx_full, vy_full
    elif vfile is not None:
        case = 2
        v = clip(vfile, geoms)
        v = v[v > -9998]  # remove NaN points
    else:
        case = 0
        raise TypeError('Either vfile or vxfile+vyfile are required.')
  
    if case == 1:
        if wfile is not None:
            xy_full = np.vstack([vx_full, vy_full, w_full])
        else:
            xy_full = np.vstack([vx_full, vy_full])
        
        if len(vx_full) > max_n:
            ## See https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy.random.Generator.choice
            rng = np.random.default_rng()
            xy = rng.choice(xy_full, size=max_n, replace=False, axis=1)
        else:
            xy = xy_full
        
        if wfile is not None:
            w = xy[2, :]
            w = np.where(w < 0, 0, w)
            kernel = gaussian_kde(xy[:2, :], weights=w)
            z = kernel(xy[:2, :])
        else:
            kernel = gaussian_kde(xy)
            z = kernel(xy)
            
        vx = xy[0, :]
        vy = xy[1, :]
            
        thres_multiplier = np.e ** (thres_sigma ** 2 / 2)   # normal dist., +- sigma number 
        thres = max(z) / thres_multiplier
        thres_idx = z >= thres
        idx = thres_idx    # alias

        if plot:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                
            viridis = cm.get_cmap('viridis', 12)
            pt_style = {'s': 6, 'edgecolor': None}
            
            if vx_full is not None:
                ax.scatter(vx_full, vy_full, color='xkcd:gray', alpha=0.2, **pt_style)
            
            ax.scatter(vx[idx], vy[idx], c=z[idx], **pt_style)
            ax.scatter(vx[~idx], vy[~idx], color=viridis(0), alpha=0.4, **pt_style)
            
        return vx, vy, z, thres_idx
    
    elif case == 2:
        
        if plot:
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                
        bins = ax.hist(v ** 2, 100);
        return v, bins

def plot_off_ice_errors(vx, vy, z, thres_idx, ax=None, zoom=True):
    
    viridis = cm.get_cmap('viridis', 12)
    pt_style = {'s': 6, 'edgecolor': None}
    idx = thres_idx    # alias
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
    ax.scatter(vx[idx], vy[idx], c=z[idx], **pt_style)
    ax.scatter(vx[~idx], vy[~idx], color=viridis(0), alpha=0.4, **pt_style)
    if zoom:
        ax.set_xlim((min(vx[idx]), max(vx[idx])))
        ax.set_ylim((min(vy[idx]), max(vy[idx])))
    
def sobel_scattering(vxfile=None, vyfile=None, wfile=None, on_ice_area=None, thres_sigma=3.0, plot=True, ax=None, max_n=10000, max_s=100):
    """

    """ 
    shapefile = gpd.read_file(on_ice_area)
    geoms = shapefile.geometry.values
    geoms = [mapping(geoms[i]) for i in range(len(geoms))]
    
    # def clip(gtiff, geoms):
    #     with rasterio.open(gtiff) as src:
    #         out_image, out_transform = mask(src, geoms, crop=True, nodata=-9999.0)
    #     try:
    #         clipped_data = out_image.data[0]
    #     except NotImplementedError:
    #         clipped_data = out_image[0]
    #     return clipped_data
    
    with rasterio.open(vxfile) as srcx, rasterio.open(vyfile) as srcy:
        vx_full = srcx.read(1)
        vy_full = srcy.read(1)
    if wfile is not None:
        with rasterio.open(wfile) as srcw:
            w_full = srcw.read(1)
    else:
        w_full = None
        
    nonNaN_pts_idx = np.logical_and(vx_full > -9998, vy_full > -9998)
    vx_full[~nonNaN_pts_idx] = np.nan  # replace NaN points with np.nan
    vy_full[~nonNaN_pts_idx] = np.nan  # replace NaN points with np.nan
        
    sxx = sobel(vx_full,axis=0,mode='constant')
    sxy = sobel(vx_full,axis=1,mode='constant')
    # Get square root of sum of squares
    sobelx = np.hypot(sxx,sxy)

    syx = sobel(vy_full,axis=0,mode='constant')
    syy = sobel(vy_full,axis=1,mode='constant')
    # Get square root of sum of squares
    sobely = np.hypot(syx,syy)
    
    # vx_full = None
    # vy_full = None
    # w_full = None
    
    sx_full = mask_by_shp(shapefile['geometry'], sobelx, rasterio.open(vxfile))
    sy_full = mask_by_shp(shapefile['geometry'], sobely, rasterio.open(vyfile))

    sobel_NaN_pts_idx     = np.logical_or(np.isnan(sx_full), np.isnan(sy_full))
    sobel_outlier_pts_idx = np.logical_or(sx_full > max_s, sy_full > max_s)
    sobel_bad_pts_idx = np.logical_or(sobel_NaN_pts_idx, sobel_outlier_pts_idx)
    sx_full = sx_full[~sobel_bad_pts_idx]
    sy_full = sy_full[~sobel_bad_pts_idx]

    # return sx_full, sy_full
    
    if w_full is not None:
        w_full = w_full[~sobel_bad_pts_idx]
        xy_full = np.vstack([sx_full, sy_full, w_full])
    else:
        xy_full = np.vstack([sx_full, sy_full])
        
    # return xy_full

    if len(sx_full) > max_n:
        ## See https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy.random.Generator.choice
        rng = np.random.default_rng()
        xy = rng.choice(xy_full, size=max_n, replace=False, axis=1)
    else:
        xy = xy_full
    
    # return xy

    if wfile is not None:
        w = xy[2, :]
        w = np.where(w < 0, 0, w)
        kernel = gaussian_kde(xy[:2, :], weights=w)
        z = kernel(xy[:2, :])
    else:
        kernel = gaussian_kde(xy)
        z = kernel(xy)
    
    
    # xy_full = np.vstack([sx_full.flatten(), sy_full.flatten()])
    
    sx = xy[0, :]
    sy = xy[1, :]
            
    thres_multiplier = np.e ** (thres_sigma ** 2 / 2)   # normal dist., +- sigma number 
    thres = max(z) / thres_multiplier
    thres_idx = z >= thres
    idx = thres_idx    # alias

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        viridis = cm.get_cmap('viridis', 12)
        pt_style = {'s': 6, 'edgecolor': None}

        if sx_full is not None:
            ax.scatter(sx_full, sy_full, color='xkcd:gray', alpha=0.2, **pt_style)

        ax.scatter(sx[idx], sy[idx], c=z[idx], **pt_style)
        ax.scatter(sx[~idx], sy[~idx], color=viridis(0), alpha=0.4, **pt_style)

    return sx, sy, z, thres_idx
    

    
    

    
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
    
def mask_by_shp(geom,array,ds):
    """
    retrive date from input raster array falling within input polygon
    
    Parameters
    -------------
    geom: shapely.geometry 
        shapefile within which to return raster values
        if using geopandas GeoDataFrame, input should be geometry column like gdf['geometry']
    array: np.ma.array
        masked array of input raster
    ds: gdal or rasterio dataset
        dataset information for input array
        used in computing geotransform
    
    Returns
    -------------
    masked_array: np.ma.array
        input array containing non-masked values for only regions falling within input geometry
    """
    
    from rasterio import features
 
    if (type(ds) == rasterio.io.DatasetReader):
        transform = ds.transform
    else:
        from affine import Affine
        transform = Affine.from_gdal(*ds.GetGeoTransform())
    shp = features.rasterize(geom,out_shape=np.shape(array),fill=-9999,transform=transform,dtype=float)
    shp_mask = np.ma.masked_where(shp==-9999,shp)
    masked_array = np.ma.array(array,mask=shp_mask.mask)
    return masked_array


def compute_buffer(shp,buffer_dist=500,external_only=True):
    """
    Create a buffer along an input polygon shapefile

    Parameters
    ------------
    shp: gpd.GeoDataFrame
        object containing the polygons to be buffered 
        (preferred projection crs should be metric (like UTM) and not geographic (in degrees)
    buffer_dist: numeric
        distance till which buffer will be computed (units will depend on shapefile CRS (m for metric projections)
    external_only: bool (default: True)
        return shapefile containing buffer from outside the polygon area only
        useful for cases when computing static area stats about glacier boundaries
    
    Returns
    ------------
    out_shp: gpd.GeoDataFrame
        buffered shapefile
    """
    
    shp_buffer = shp.copy()
    shp_buffer['geometry'] = shp_buffer['geometry'].buffer(buffer_dist)
    if external_only:
        shp_buffer_external = gpd.overlay(shp_buffer,shp,how='difference')
        out_shp = shp_buffer_external
    else:
        out_shp = shp_buffer
    return out_shp
