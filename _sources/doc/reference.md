# Reference

## `glaft.Velocity` class

### `glaft.Velocity(vxfile: str=None, vyfile: str=None, wfile: str=None, static_area: str=None, on_ice_area: str=None, nodata: float=-9999.0, velocity_unit: str='m/day', thres_sigma: float=2.0, kde_gridsize: int=60)`

Construct an experiment to calculate velocity map benchmarking metrics.

    vxfile:        str, geotiff file path (must contain only one band and use meters as the geotransform unit)
    vyfile:        str, geotiff file path (must contain only one band and use meters as the geotransform unit)
    wfile:         str, goetiff file path (as weight)
    static_area:   str, static area (shapefile) file path
    on_ice_area:   str, on-ice area (shapefile) file path
    nodata:      float, nodata value in the provided geotiff. NOT FULLY IMPLEMENTED YET.
    velocity_unit: str, velocity unit to be shown on the result plots.
    thres_sigma: float, selected thresholding z values.
    kde_gridsize:  int, grid size used for evaluating a crude KDE surface (larger value means a faster but less precise process)
    
#### `Velocity.static_terrain_analysis(plot=None, ax=None)`

Perform the static terrain analysis and calculate the correct-match uncertainty. 

    plot: None for no plot; "full" for plotting the results in the full extent; "zoomed" for plotting the results in the zoomed extent.
    ax: matplotlib.axes object to be visualized when plot is not set to None.
    
#### `Velocity.longitudinal_shear_analysis(plot=None, ax=None)`

Perform the longitudinal strain rate analysis and calculate the associated metrics. 

    plot: None for no plot; "full" for plotting the results in the full extent; "zoomed" for plotting the results in the zoomed extent.
    ax: matplotlib.axes object to be visualized when plot is not set to None.

#### `Velocity.plot_full_extent(ax=None, rect=None, metric: int=1, **pt_style)`

Plot the analysis results in the full extent.

    ax: matplotlib.axes object to be visualized when plot is not set to None.
    rect: style of the rectanglar box indicating the correct match area. If None, the plot uses thick red line as default. See Velocity.create_rectangle_patch
    metric: which metric to be plotted. 1: static terrain; 2: along-flow strain rate.
    pt_style: point style passed to plt.scatter.

#### `Velocity.plot_zoomed_extent(self, ax=None, rect=None, base_colormap=None, metric: int=1, **pt_style)`

Plot the analysis results in the zoomed extent (focusing on the correct-match area).

    ax: matplotlib.axes object to be visualized when plot is not set to None.
    rect: style of the rectanglar box indicating the correct match area. If None, the plot uses thick red line as default. See Velocity.create_rectangle_patch
    base_colormap: colormap for showing the KDE probability distribtuion. If None, the default (cramericm.bamako_r) is used. 
    metric: which metric to be plotted. 1: static terrain; 2: along-flow strain rate.
    pt_style: point style passed to plt.scatter.

#### `Velocity.create_rectangle_patch(var_x, var_y, **rect_style)`

Create a rectangle patch to be plotted on the visualization of the processing results.

    var_x: half width of the rectangle
    var_y: half height of the rectangle
    rect_style: style of the rectangle. Default is {'linewidth': 2, 'edgecolor': 'xkcd:cranberry', 'facecolor': 'none', 'alpha': 0.7}

    Returns
    --------------
    matplotlib.patches.Rectangle object.

#### `Velocity.plot_strain_map(ax=None, base_colormap=None, vmax=None)`

Show the strain map (only works after `longitudinal_shear_analysis` is performed).

    ax: plotting axes (matplotlib.axes object)
    base_colormap: colormap for showing the strain map. If None, the default (cramericm.bamako) is used. 
    vmax: value at where color is saturated. If None, this upper limit will be automatically scaled.

#### `Velocity.cal_invalid_pixel_percent()`

Calculate the amount of invalid pixels in a velocity map.

#### Important attributes

- `Velocity.vxfile` Vx file path.
- `Velocity.vyfile` Vy file path.
- `Velocity.wfile` Weight file path.
- `Velocity.static_area` static area geometries file path. 
- `Velocity.on_ice_area` on-ice area geometries file path.
- `Velocity.nodata` NoData value for the raster files.
- `Velocity.velocity_unit` velocity unit to be shown on the result plots.
        
- `Velocity.vx`  2-D, clipped vx data
- `Velocity.vy`  2-D, clipped vy data
- `Velocity.xy`  = np.vstack([flattened_vx, flattened_vy]) --> (1-D)-by-2 array containing vx and vy. 
- `Velocity.w`   2-D, clipped weight data
- `Velocity.w_flat` 1-D, clipped & flattened weight data
- `Velocity.dx` velocity map pixel spacing, x
- `Velocity.dy` velocity map pixel spacing, y
- `Velocity.thres_sigma` selected thresholding z values.
        
- `Velocity.kernel` KDE kernel (default `epanechnikov`)
- `Velocity.bandwidth` KDE bandwidth (default: calculated using the rule of thumb)
- `Velocity.kde_gridsize` grid size used for evaluating a crude KDE surface (larger value means a faster but less precise process)
- `Velocity.mesh_fine` Final KDE mesh
- `Velocity.mesh_fine_z` KDE values at the final KDE mesh vertices
- `Velocity.mesh_fine_thres_idx` boolean array showing whether a vertex falls within the correct match area
- `Velocity.metric_static_terrain_x` delta_x
- `Velocity.metric_static_terrain_y` delta_y
        
- `Velocity.kdepeak_x` KDE peak location x
- `Velocity.kdepeak_y` KDE peak location y
- `Velocity.outlier_percent` Incorrect match percentage (*100%)
- `Velocity.invalid_percent` Invalid pixels percentage (*100%)
                
- `Velocity.flow_theta` Flow direction
- `Velocity.exx` normal strain rate exx, image axis
- `Velocity.eyy` normal strain rate eyy, image axis
- `Velocity.exy` shear strain rate exy, image axis
- `Velocity.flow_exx` normal strain rate exx, flow axis
- `Velocity.flow_eyy` normal strain rate eyy, flow axis
- `Velocity.flow_exy` shear strain rate exy, flow axis
- `Velocity.metric_alongflow_normal` delta_x'x'
- `Velocity.metric_alongflow_shear` delta_x'y'

## Auxillary functions

### `glaft.show_velocomp(gtiff: str, ax=None, **cm_settings)`

Preview a geotiff file (presuming a velocity map).

    Parameters
    -------------
    gtiff: geotiff file path (much be single band. e.g., Velocity component, such as Vx)
    ax: plotting axes (matplotlib.axes object)
    cm_settings: colormap settings to be used in the visualization. Default: {'vmin': -2, 'vmax': 2, 'cmap': cramericm.broc_r}
    
    Returns
    --------------
    cm_settings: colormap settings using in the visualization.
        image with NAOF filtered applied
       
### `glaft.prep_colorbar_mappable(vmin=-2, vmax=2, cmap=cramericm.broc_r)`

Generate a mappbale object for creating a colorbar.

    Parameters
    -------------
    vmin: colormap lower range
    vmax: colormap upper range
    cmap: master colormap (a colormap object)
    
    Returns
    --------------
    mappable: a matplotlib.mappable object
        
### `glaft.naof2(im)`

NAOF preprocessing filter. Translated to Python from GIV source code. All implementation credit to [Max VWDV]( https://github.com/MaxVWDV/glacier-image-velocimetry/blob/16fb4d2d243b6dc24f35c531d3ea8d91bf3c84a4/functions/image%20processing%20and%20filters/NAOF2.m). 

MaxVWDV. (2021). MaxVWDV/glacier-image-velocimetry: Glacie Image Velocimetry (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.4904544
    
    Parameters
    -------------
    im: np.ma.array
        input image
    
    Returns
    --------------
    naof_im: np.ma.array
        image with NAOF filtered applied