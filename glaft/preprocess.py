#! /usr/bin/env python

import numpy as np
from cmcrameri import cm as cramericm
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import matplotlib as mpl

def naof2(im):
    """
    NAOF preprocessing filter 
    Translated to python from GIV source code
    All implementation credit to Max VWDV (https://github.com/MaxVWDV/glacier-image-velocimetry/blob/16fb4d2d243b6dc24f35c531d3ea8d91bf3c84a4/functions/image%20processing%20and%20filters/NAOF2.m)
    
    MaxVWDV. (2021). MaxVWDV/glacier-image-velocimetry: Glacie Image Velocimetry (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.4904544
    
    Parameters
    -------------
    im: np.ma.array
        input image
    
    Returns
    --------------
    naof_im: np.ma.array
        image with NAOF filtered applied
    """
    
    # Create filter bank
    filter_1 = np.array([-1, 2, -1]).reshape((3,1))
    filter_2 = np.array([-1, 2, -1]).reshape((1,3))
    filter_3 = np.array([[-1, 0, 0],[0,2,0], [0,0,-1]])
    filter_4 = np.array([[0, 0, -1],[0, 2, 0],[-1, 0, 0]])
    
    # creation of filtered images
    from scipy.signal import convolve2d
    # there might be some differences on the edges, see here (https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function)
    filt1 = convolve2d(im,filter_1,'same')
    filt2 = convolve2d(im,filter_2,'same')
    filt3 = convolve2d(im,filter_3,'same')
    filt4 = convolve2d(im,filter_4,'same')
    
    # 2 argument arctan
    at1 = np.arctan2(filt1,filt2)
    at2 = np.arctan2(filt3,filt4)
    
    # go back to orinal feature space
    naof_im = np.cos(at1) + np.cos(np.pi/2 - at1) + np.cos(at2) + np.cos(np.pi/2 - at2)
    
    return naof_im

def show_velocomp(gtiff: str, ax=None, **cm_settings):
    """
    gtiff: str, geotiff file name (velocity component, such as Vx)
    """
    
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
    if not cm_settings:
         cm_settings = {'vmin': -2, 'vmax': 2, 'cmap': cramericm.broc_r}
        
    with rasterio.open(gtiff) as src:
        show(src, ax=ax, **cm_settings)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    return cm_settings
    
def prep_colorbar_mappable(vmin=-2, vmax=2, cmap=cramericm.broc_r):
    """
    generate a mappbale object for creating a colorbar.
    """
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return mappable
