import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import map_coordinates, sobel
# from scipy.signal import medfilt
from scipy.signal import medfilt2d
import geopandas as gpd
from shapely.geometry import mapping
import rasterio
from rasterio.mask import mask
from rasterio import features
import matplotlib.pyplot as plt
from matplotlib import cm
from cmcrameri import cm as cramericm
import warnings
from sklearn.neighbors import KernelDensity
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from functools import wraps

def _log_method(clsmethod):
    @wraps(clsmethod)
    def wrapper(*args, **kwargs):
        print('Running {}'.format(clsmethod.__name__))
        clsmethod(*args, **kwargs)
    return wrapper


class Velocity():
    
    def __init__(self, 
                 vxfile:        str=None,
                 vyfile:        str=None,
                 wfile:         str=None,
                 static_area:   str=None,
                 on_ice_area:   str=None,
                 nodata:      float=-9999.0,
                 velocity_unit: str='m/day',
                 thres_sigma: float=2.0,
                 kde_gridsize:  int=60,
                ):
        """
        vxfile:        str, geotiff file path (must contain only one band and use meters as the geotransform unit)
        vyfile:        str, geotiff file path (must contain only one band and use meters as the geotransform unit)
        wfile:         str, goetiff file path (as weight)
        static_area:   str, static area (shapefile) file path
        on_ice_area:   str, on-ice area (shapefile) file path
        nodata:      float, nodata value in the provided geotiff. NOT FULLY IMPLEMENTED YET.
        velocity_unit: str, velocity unit to be shown on the result plots.
        """ 
        self.vxfile = vxfile
        self.vyfile = vyfile
        self.wfile = wfile
        self.static_area = static_area
        self.on_ice_area = on_ice_area
        self.nodata = nodata
        self.velocity_unit = velocity_unit
        
        self.vx = None                  # 2-D, clipped vx
        self.vy = None                  # 2-D, clipped vy
        self.xy = None                  # np.vstack([flattened_vx, flattened_vy]) --> (1-D)-by-2 
        self.w = None                   # 2-D, clipped weight data
        self.w_flat = None              # 1-D, clipped & flattened weight data
        self.dx = None
        self.dy = None
        self.thres_sigma = thres_sigma
        
        self.kernel = 'epanechnikov'
        self.kde = None
        self.bandwidth = None
        self.xystd = None
        self.kde_gridsize = kde_gridsize
        self.mesh_crude = None
        self.mesh_crude_shape = None
        self.mesh_crude_z = None
        self.mesh_fine  = None
        self.mesh_fine_shape  = None
        self.mesh_fine_z = None
        self.mesh_fine_thres_idx = None
        self.metric_static_terrain_x = None
        self.metric_static_terrain_y = None
        
        self.kdepeak_x = None
        self.kdepeak_y = None
        self.outlier_percent = None
        self.invalid_percent = None
                
        self.flow_theta = None
        self.exx = None
        self.eyy = None
        self.exy = None
        self.flow_exx = None
        self.flow_eyy = None
        self.flow_exy = None
        self.metric_alongflow_normal = None
        self.metric_alongflow_shear = None
        
        
    @staticmethod       
    def _clip(gtiff: str, shp: str, nodata: float=-9999.0):
        """
        gtiff: geotiff file path
        shp: polygon shapefile file path
        nodata: nodata value defined in the geotiff.
        ----
        returns:
        clipped_data: 1-D array containing valid pixel values within the polygon shapes. 
        """
        shapes = gpd.read_file(shp)
        geoms = shapes.geometry.values
        geoms = [mapping(geoms[i]) for i in range(len(geoms))]
        with rasterio.open(gtiff) as src:
            out_image, out_transform = mask(src, geoms, crop=False, nodata=nodata)    # crop=True
        try:
            clipped_data = out_image.data[0]
        except NotImplementedError:
            clipped_data = out_image[0]
        return clipped_data
    
    @staticmethod
    def _construct_kde_eval_mesh(midx:      float, 
                                 midy:      float,
                                 halfwidth: float, 
                                 gridsize:  int
                                ):
        """

        ----
        returns:
        1. xyeval: n-by-2 array; each row is an x-y coordinate to be evaluated for KDE.
        2. The shape of the evaluating grid.
        """
        xeval = np.linspace(midx - halfwidth, midx + halfwidth, gridsize)
        yeval = np.linspace(midy - halfwidth, midy + halfwidth, gridsize)
        xeval_grid, yeval_grid = np.meshgrid(xeval, yeval)
        xyeval = np.vstack([xeval_grid.flatten(), yeval_grid.flatten()]).T
        return xyeval, np.shape(xeval_grid)
    
    @staticmethod
    def _verify_axes(ax):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        return ax
    
    @staticmethod
    def _customize_colormap(cmap: ListedColormap):
        """
        Add a transparency setting to cmap.
        """
        custom_cmap = cmap(np.linspace(0, 1, 254))
        custom_cmap[:, 3] = np.linspace(0, 1, 254)    # alpha channel (trasparent --> non-transparent)
        return ListedColormap(custom_cmap)
    
    @_log_method
    def clip_static_area(self):
        """
        self.xy: 2-by-n array.
        """
        
        if self.vxfile is None or self.vyfile is None:
            raise TypeError('Vxfile and Vyfile are required.')
            
        if int(self.nodata) != -9999:
            warnings.warn("NoData values other than -9999 has not implemented yet. Falling back to -9999.")
            nodata_val = -9999.0
        else:
            nodata_val = self.nodata
            
        self.vx = self._clip(self.vxfile, self.static_area, nodata=nodata_val)
        self.vy = self._clip(self.vyfile, self.static_area, nodata=nodata_val)
        
        nonNaN_pts_idx = np.logical_and(self.vx > -9998, self.vy > -9998)
        vx = self.vx[nonNaN_pts_idx]  # remove NaN points; flatten the array as well
        vy = self.vy[nonNaN_pts_idx]  # remove NaN points; flatten the array as well
        
        if self.wfile is not None:
            self.w = self._clip(self.wfile, self.static_area, nodata=nodata_val)
            w = self.w[nonNaN_pts_idx]  # remove NaN points
            w = np.where(w < 0, 0, w)  # force negative w to zero
            self.w_flat = w
            
        self.xy = np.vstack([vx, vy])

    @_log_method
    def calculate_xystd(self):
        
        if self.xy is None:
            raise TypeError('There must be selected pixels.')
            
        xycov = np.cov(self.xy)
        self.xystd = (xycov[0, 0] * xycov[1, 1]) ** (0.25)
        
    @_log_method
    def calculate_bandwidth(self):
        """
        Calculate the rule-of-thumb bandwidth for the selected kernel,
        according to https://doi.org/10.1016/j.spl.2012.07.020 
        For now, only the epanechnikov and gaussian kernels are implemented.
        """
        
        if self.xy is None or self.xystd is None:
            raise TypeError('Needs to set up self.xy and self.xystd first.')
            
        if self.kernel == 'epanechnikov':
            self.bandwidth = 2.1991 * self.xystd * self.xy.shape[1] ** (-1. / (2 + 4))
        elif self.kernel == 'gaussian':
            self.bandwidth = 1.0000 * self.xystd * self.xy.shape[1] ** (-1. / (2 + 4))
        else:
            raise NotImplementedError('Only Epanechnikov and Gaussian kernels are avaliable for now.')
        # self.bandwidth = 0.4   # testing
        
    @_log_method
    def construct_crude_mesh(self):
        
        if self.xy is None or self.xystd is None:
            raise TypeError('Needs to set up self.xy and self.xystd first.')
            
        midx = np.median(self.xy[0, :])
        midy = np.median(self.xy[1, :])
        halfwidth = (self.thres_sigma + 1) * self.xystd
        
        self.mesh_crude, self.mesh_crude_shape = self._construct_kde_eval_mesh(midx, midy, halfwidth, self.kde_gridsize)
        
    @_log_method
    def calculate_kde(self):
        
        if self.bandwidth is None:
            raise TypeError('Needs to define bandwidth. Run calculate_bandwidth first.')
            
        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(self.xy.T, sample_weight=self.w_flat)
        
    @_log_method
    def eval_crude_mesh(self):
        
        log_density = self.kde.score_samples(self.mesh_crude)
        self.mesh_crude_z = np.exp(log_density)
        
    @_log_method
    def construct_fine_mesh(self):
        
        if self.mesh_crude_z is None:
            raise TypeError('Needs to calculate self.mesh_crude_z first.')
            
        midx = np.median(self.xy[0, :])
        midy = np.median(self.xy[1, :])
        
        thres_multiplier = np.e ** (self.thres_sigma ** 2 / 2)   # normal dist., +- sigma number 
        thres_value = max(self.mesh_crude_z) / thres_multiplier
        thres_idx = self.mesh_crude_z >= thres_value
        vertice_x = self.mesh_crude[:, 0]
        vertice_y = self.mesh_crude[:, 1]
        refined_mesh_spacing = ((max(vertice_x[thres_idx]) - min(vertice_x[thres_idx])) + (max(vertice_y[thres_idx]) - min(vertice_y[thres_idx]))) / 2
        halfwidth = refined_mesh_spacing
        
        self.mesh_fine, self.mesh_fine_shape = self._construct_kde_eval_mesh(midx, midy, halfwidth, self.kde_gridsize)
        
    @_log_method  
    def eval_fine_mesh(self):
        
        log_density = self.kde.score_samples(self.mesh_fine)
        self.mesh_fine_z = np.exp(log_density)
        
    @_log_method
    def thresholding_fine_mesh(self):
        
        if self.mesh_fine_z is None:
            raise TypeError('Needs to calculate self.mesh_fine_z first.')
            
        thres_multiplier = np.e ** (self.thres_sigma ** 2 / 2)   # normal dist., +- sigma number 
        thres_value = max(self.mesh_fine_z) / thres_multiplier
        self.mesh_fine_thres_idx = self.mesh_fine_z >= thres_value
        
    @_log_method
    def thresholding_metric(self, metric: int=1):
        
        vertice_x = self.mesh_fine[:, 0]
        vertice_y = self.mesh_fine[:, 1]

        self.kdepeak_x = vertice_x[np.argmax(self.mesh_fine_z)]
        self.kdepeak_y = vertice_y[np.argmax(self.mesh_fine_z)]
        
        if metric == 1:
            self.metric_static_terrain_x = (max(vertice_x[self.mesh_fine_thres_idx]) - min(vertice_x[self.mesh_fine_thres_idx])) / 2
            self.metric_static_terrain_y = (max(vertice_y[self.mesh_fine_thres_idx]) - min(vertice_y[self.mesh_fine_thres_idx])) / 2
        elif metric == 2:
            self.metric_alongflow_normal = (max(vertice_x[self.mesh_fine_thres_idx]) - min(vertice_x[self.mesh_fine_thres_idx])) / 2
            self.metric_alongflow_shear  = (max(vertice_y[self.mesh_fine_thres_idx]) - min(vertice_y[self.mesh_fine_thres_idx])) / 2
            
    @_log_method
    def cal_outlier_percent(self):
        
        vertice_x = self.mesh_fine[:, 0]
        vertice_y = self.mesh_fine[:, 1]
        
        x_lb = min(vertice_x[self.mesh_fine_thres_idx])
        x_ub = max(vertice_x[self.mesh_fine_thres_idx])
        y_lb = min(vertice_y[self.mesh_fine_thres_idx])
        y_ub = max(vertice_y[self.mesh_fine_thres_idx])
        
        x = self.xy[0, :]
        y = self.xy[1, :]
        
        x_outside_idx = np.logical_or(x < x_lb, x > x_ub)
        y_outside_idx = np.logical_or(y < y_lb, y > y_ub)
        xy_outside_idx = np.logical_or(x_outside_idx, y_outside_idx)
        
        self.outlier_percent = np.sum(xy_outside_idx) / x.size
        
            

        
    def create_rectangle_patch(self, var_x, var_y, **rect_style):
        
        vertice_x = self.mesh_fine[:, 0]
        vertice_y = self.mesh_fine[:, 1]
        
        if not rect_style:
            rect_style = {'linewidth': 2, 'edgecolor': 'xkcd:cranberry', 'facecolor': 'none', 'alpha': 0.7}            
        rect = patches.Rectangle((min(vertice_x[self.mesh_fine_thres_idx]), min(vertice_y[self.mesh_fine_thres_idx])), 
                                 2 * var_x,
                                 2 * var_y, 
                                 **rect_style)
        return rect
    
    def gridding_kde_results(self):
        
        xg = np.reshape(self.mesh_fine[:, 0], self.mesh_fine_shape)
        yg = np.reshape(self.mesh_fine[:, 1], self.mesh_fine_shape)
        zg = np.reshape(self.mesh_fine_z, self.mesh_fine_shape)
        return xg, yg, zg
    
    def _determine_plot_variables(self, metric: int=1):
        """
        See self.plot_full_extent for explanation.
        """
        if metric == 1:
            var_x = self.metric_static_terrain_x
            var_y = self.metric_static_terrain_y
            lbl_x = '$\delta_u$'
            lbl_y = '$\delta_v$'
            unit = self.velocity_unit
        elif metric == 2:
            var_x = self.metric_alongflow_normal
            var_y = self.metric_alongflow_shear
            lbl_x = "$\delta_{x'x'}$"
            lbl_y = "$\delta_{x'y'}$"
            unit = '1/' + self.velocity_unit.split('/')[-1]
        else:
            raise NotImplementedError('Metric can only be 1 (static terrain) or 2 (along-flow strain rate).')
            
        return var_x, var_y, lbl_x, lbl_y, unit
        
    def plot_full_extent(self, ax=None, rect=None, metric: int=1, **pt_style):
        """
        metric: which metric to be plotted.
            1: static terrain
            2: along-flow strain rate
        """
        
        ax = self._verify_axes(ax)
        
        if not pt_style:
            pt_style = {        's': 6, 
                            'color': 'xkcd:gray', 
                        'edgecolor': None, 
                            'alpha': 0.5}
        ax.scatter(self.xy[0, :], self.xy[1, :], **pt_style)
        ax.axis('equal')
        
        var_x, var_y, lbl_x, lbl_y, unit = self._determine_plot_variables(metric=metric)
        
        if var_x:
            
            if not rect:
                rect = self.create_rectangle_patch(var_x, var_y)
            ax.add_patch(rect)
            ax.set_title('{} = {:6.3f} | {} = {:6.3f} ({})'.format(lbl_x,
                                                                   var_x,
                                                                   lbl_y,
                                                                   var_y,
                                                                   unit))
            
    def plot_zoomed_extent(self, ax=None, rect=None, base_colormap=None, metric: int=1, **pt_style):
        """
        metric: which metric to be plotted.
            1: static terrain
            2: along-flow strain rate
        """
        
        ax = self._verify_axes(ax)
        
        if not base_colormap:
            base_colormap = cramericm.bamako_r
            # other good-looking colormaps:
            # base_colormap = cramericm.turku_r
            # base_colormap = cramericm.hawaii_r
            # base_colormap = cramericm.batlow_r
        
        if not pt_style:
            pt_style = {        's': 1, 
                            'color': base_colormap(0), 
                        'edgecolor': None, 
                            'alpha': 0.5}
        ax.scatter(self.xy[0, :], self.xy[1, :], **pt_style)
        ax.axis('equal')
        
        var_x, var_y, lbl_x, lbl_y, unit = self._determine_plot_variables(metric=metric)
        
        if var_x:
        
            xg, yg, zg = self.gridding_kde_results()
            custom_cmap = self._customize_colormap(base_colormap)

            ax.pcolormesh(xg, yg, zg, shading='nearest', cmap=custom_cmap)
            ax.set_xlim(np.min(xg), np.max(xg))
            ax.set_ylim(np.min(yg), np.max(yg))
            
            if not rect:
                rect = self.create_rectangle_patch(var_x, var_y)
            ax.add_patch(rect)
            ax.set_title('{} = {:6.3f} | {} = {:6.3f} ({})'.format(lbl_x,
                                                                   var_x,
                                                                   lbl_y,
                                                                   var_y,
                                                                   unit))
        else:
            warnings.warn("Unable to zoom in: need KDE estimates to determine zoom-in range.")
            
            
    def static_terrain_analysis(self, plot=None, ax=None):
        """
        Entire workflow.
        """
        
        metric = 1
        self.clip_static_area()
        self.calculate_xystd()
        self.calculate_bandwidth()
        self.calculate_kde()
        self.construct_crude_mesh()
        self.eval_crude_mesh()
        self.construct_fine_mesh()
        self.eval_fine_mesh()
        self.thresholding_fine_mesh()
        self.thresholding_metric(metric=metric)
        self.cal_outlier_percent()
        
        if plot == 'full':
            self.plot_full_extent(ax=ax, metric=metric)
        elif plot == 'zoomed':
            self.plot_zoomed_extent(ax=ax, metric=metric)
            
            
    @staticmethod
    def _calculate_strain_rate(xx: np.array, yy: np.array, dx: float=1., dy: float=1.):
        """
        xx: 2-D Vx grid (Cartesian convention; Vx[0, 0] is the lower-left corner)
        yy: 2-D Vy grid (Cartesian convention; Vy[0, 0] is the lower-left corner)
        dx: grid spacing along x
        dy: grid spacing along y
        
        Sobel kernel to be convoluted with xx:
        | 1 0 -1 |
        | 2 0 -2 |
        | 1 0 -1 |
        
        Sobel kernel to be convoluted with yy:
        |  1  2  1 |
        |  0  0  0 |
        | -1 -2 -1 |
        
        Edge mode is set to constant, padding with np.nan. 
        """
        # These four are pixel-based gradient. (the same unit from velocity)
        duxdx = sobel(xx, axis=1, mode='constant', cval=np.nan)
        duxdy = sobel(xx, axis=0, mode='constant', cval=np.nan)
        duydx = sobel(yy, axis=1, mode='constant', cval=np.nan)
        duydy = sobel(yy, axis=0, mode='constant', cval=np.nan)
        
        # Strain rate = pixel-based gradient divied by pixel spacing. It has the unit of (Velocity * L^-1).
        exx = duxdx / dx                         # normal strain rate along x axis
        eyy = duydy / dy                         # normal strain rate along y axis
        exy = 0.5 * (duxdy / dy + duydx / dx)    #  shear strain rate at the x-y coordinates
        
        return exx, eyy, exy

    @staticmethod
    def _rotate_strain_rate(exx: np.array, eyy: np.array, exy: np.array, theta: [float, np.array]):
        """
        theta can be a single float value or an array with the same size of exx, eyy, and exy.
        """
        
        exx_rot = exx * np.cos(theta) ** 2 + eyy * np.sin(theta) ** 2 + exy * np.sin(2 * theta)
        eyy_rot = exx * np.sin(theta) ** 2 + eyy * np.cos(theta) ** 2 - exy * np.sin(2 * theta)
        exy_rot = 0.5 * (eyy - exx) * np.sin(2 * theta) +  exy * np.cos(2 * theta)
        return exx_rot, eyy_rot, exy_rot
    
    @staticmethod
    def _principle_strain_rate(exx: np.array, eyy: np.array, exy: np.array):
        """
        find the principle strain rates (e1 & e2) and their rotating angle from exx and eyy. 
        """
        
        principle_theta = 0.5 * np.arctan(2 * exy / (exx - eyy))
        e1 = 0.5 * (exx + eyy) + (exy ** 2 + 0.25 * (exx - eyy) ** 2 ) ** 0.5
        e2 = 0.5 * (exx + eyy) - (exy ** 2 + 0.25 * (exx - eyy) ** 2 ) ** 0.5
        return e1, e2, principle_theta


    @_log_method
    def clip_on_ice_area(self):
        """

        """
        
        if self.vxfile is None or self.vyfile is None:
            raise TypeError('Vxfile and Vyfile are required.')
            
        if int(self.nodata) != -9999:
            warnings.warn("NoData values other than -9999 has not implemented yet. Falling back to -9999.")
            nodata_val = -9999.0
        else:
            nodata_val = self.nodata
            
        self.vx = self._clip(self.vxfile, self.on_ice_area, nodata=nodata_val)
        self.vy = self._clip(self.vyfile, self.on_ice_area, nodata=nodata_val)
        
            
        nonNaN_pts_idx = np.logical_and(self.vx > -9998, self.vy > -9998)
        self.vx[~nonNaN_pts_idx] = np.nan  # change all NoData points to np.nan
        self.vy[~nonNaN_pts_idx] = np.nan  # change all NoData points to np.nan
        
        if self.wfile is not None:
            self.w = self._clip(self.wfile, self.on_ice_area, nodata=nodata_val)
            self.w[~nonNaN_pts_idx] = np.nan  # change all NoData points to np.nan
            self.w = np.where(self.w < 0, 0, self.w)  # force negative w to zero
            
    @_log_method
    def get_grid_spacing(self):
        """
        NOTE: We assume Vx and Vy have the SAME SIZE and do not check if this is true.
        """
        
        if self.vxfile is None or self.vyfile is None:
            raise TypeError('Vxfile and Vyfile are required.')
            
        with rasterio.open(self.vxfile) as srcx:
            transform = srcx.transform
            self.dx = transform[0]
            self.dy = abs(transform[4])

    @_log_method
    def calculate_flow_theta(self, kernel_size=None):
        """
        flow_theta is along-flow direction in radians, counterclockwise from the east (i.e. from the conventional x-axis)
            which is smoothed by a median filter with a default or provided kernel size.
        The smoothing is to prevent strain rate projected onto an erroneous flow direction, 
            which can happen if the average flow speed is low.   
        """
        
        if self.vx is None or self.vy is None:
            raise TypeError('Either Vxfile and Vyfile are required or the clipping has to be performed first.')
            
        with rasterio.open(self.vxfile) as srcx, rasterio.open(self.vyfile) as srcy:
            vx_noclip = srcx.read(1)
            vy_noclip = srcy.read(1)
            
        flow_theta = np.arctan2(vy_noclip, vx_noclip)    
        
        if not kernel_size:
            if self.dx:
                kernel_size = 1500 / self.dx
                kernel_size = int(kernel_size // 2 * 2 + 1)    # round to the closest odd integer
                kernel_size = 35 if kernel_size > 35 else kernel_size    # kernel size is capped at 35 for computing efficiency
            else:
                warnings.warn("Cannot determine kernel size based on pixel spacing. Use default kernel size = 25.")
                kernel_size = 25
                
        flow_theta = medfilt2d(flow_theta, kernel_size)
        flow_theta[np.isnan(self.vx)] = np.nan
        
        self.flow_theta = flow_theta

        
    @_log_method
    def calculate_strain_rate(self, pixel_based=False, rotate_angle: float=None):
        """
        pixel_based lets you to choose to calculate distance-based strain rate (when it sets to False) or pixel-based strain rate.
        rotate_angle lets you set up a single angle to rotate strain field. If omitted, self.flow_theta will be used.
        """
        
        # If Vx and Vy are read from geotiff, they follow the image convention; i.e., Vx[0, 0] is the upper-left corner.
        # We have to flip these array so that it follows the Cartesian convention before the Sobel operator is applied.
        vx_flipud = np.flipud(self.vx)
        vy_flipud = np.flipud(self.vy)
        
        if pixel_based:
            exx, eyy, exy = self._calculate_strain_rate(vx_flipud, vy_flipud)
        else:
            exx, eyy, exy = self._calculate_strain_rate(vx_flipud, vy_flipud, dx=self.dx, dy=self.dy)
            
        # flip the output back to the image convention, and save them to the object
        self.exx = np.flipud(exx)
        self.eyy = np.flipud(eyy)
        self.exy = np.flipud(exy)
        
        if not rotate_angle:
            rotate_angle = self.flow_theta
        
        self.exx_flow, self.eyy_flow, self.exy_flow = self._rotate_strain_rate(self.exx, self.eyy, self.exy, rotate_angle)

    @_log_method
    def prep_strain_rate_kde(self):
        """
        flatten exx_flow, eyy_flow, and w.
        """
        
        nonNaN_pts_idx = np.logical_and(~np.isnan(self.exx_flow), ~np.isnan(self.exy_flow))        
        x = self.exx_flow[nonNaN_pts_idx]
        y = self.exy_flow[nonNaN_pts_idx]
        self.xy = np.vstack([x, y])
        
        if self.w is not None:
            self.w_flat = self.w[nonNaN_pts_idx]
            
    def plot_strain_map(self, ax=None, base_colormap=None, vmax=None):
        
        ax = self._verify_axes(ax)
        
        if not base_colormap:
            base_colormap = cramericm.bamako
        
        saturation_radius = np.sqrt(self.metric_alongflow_normal * self.metric_alongflow_shear)
        along_flow_strain_mag = np.sqrt(self.exx_flow ** 2 + self.exy_flow ** 2)
        
        if not vmax:
            vmax = saturation_radius
        
        img_mappable = ax.imshow(along_flow_strain_mag, vmin=0, vmax=vmax, cmap=base_colormap)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        return img_mappable
        

    
    def longitudinal_shear_analysis(self, plot=None, ax=None):
        """
        Entire workflow.
        """
        
        metric=2
        self.clip_on_ice_area()
        self.get_grid_spacing()
        self.calculate_flow_theta()
        self.calculate_strain_rate()
        self.prep_strain_rate_kde()
        self.calculate_xystd()
        self.calculate_bandwidth()
        self.calculate_kde()
        self.construct_crude_mesh()
        self.eval_crude_mesh()
        self.construct_fine_mesh()
        self.eval_fine_mesh()
        self.thresholding_fine_mesh()
        self.thresholding_metric(metric=metric)
        self.cal_outlier_percent()
        
        if plot == 'full':
            self.plot_full_extent(ax=ax, metric=metric)
        elif plot == 'zoomed':
            self.plot_zoomed_extent(ax=ax, metric=metric)
            
            
    @_log_method
    def cal_invalid_pixel_percent(self):
        """
  
        """
        
        if self.vxfile is None or self.vyfile is None:
            raise TypeError('Vxfile and Vyfile are required.')
            
        with rasterio.open(self.vxfile) as srcx, rasterio.open(self.vyfile) as srcy:
            vx_noclip = srcx.read(1)
            vy_noclip = srcy.read(1)
            
        if np.isnan(self.nodata):
            vx_invalid_idx = np.isnan(vx_noclip)
            vy_invalid_idx = np.isnan(vy_noclip)
        elif int(self.nodata) == 0:
            vx_invalid_idx = np.abs(vx_noclip) <= np.finfo(float).eps
            vy_invalid_idx = np.abs(vy_noclip) <= np.finfo(float).eps
        else:   # e.g. -9999, -99999,...
            vx_invalid_idx = vx_noclip < self.nodata + 1
            vy_invalid_idx = vy_noclip < self.nodata + 1
            
        vxy_invalid_idx = np.logical_or(vx_invalid_idx, vy_invalid_idx)
        
        self.invalid_percent = np.sum(vxy_invalid_idx) / vxy_invalid_idx.size

        
        
############### CLASS ENDS ##############




def _clip(gtiff: str, shp: str, nodata: float=-9999.0):
    """
    gtiff: geotiff file path
    shp: polygon shapefile file path 
    """
    shapes = gpd.read_file(shp)
    geoms = shapes.geometry.values
    geoms = [mapping(geoms[i]) for i in range(len(geoms))]
    with rasterio.open(gtiff) as src:
        out_image, out_transform = mask(src, geoms, crop=True, nodata=nodata)
    try:
        clipped_data = out_image.data[0]
    except NotImplementedError:
        clipped_data = out_image[0]
    return clipped_data

def static_terrain_velo(vxfile=None, vyfile=None, wfile=None, static_area=None, thres_sigma=3.0, plot='full', ax=None, max_n=10000, peak_loc=False, nodata=-9999.0, kdegrid_size=100):
    """
    vxfile: str, geotiff file path
    vyfile: str, geotiff file path
    wfile: str, goetiff file path (as weight)
    static_area: str, static area (shapefile) file path
    max_n: maximum samples to calculate Gaussian KDE
    nodata: specify nodata value in the provided geotiff. NOT implemented yet.
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
    #     shapefile = gpd.read_file(off_ice_area)
    #     geoms = shapefile.geometry.values
    #     geoms = [mapping(geoms[i]) for i in range(len(geoms))]

    #     def clip(gtiff, geoms):
    #         with rasterio.open(gtiff) as src:
    #             out_image, out_transform = mask(src, geoms, crop=True, nodata=-9999.0)
    #         try:
    #             clipped_data = out_image.data[0]
    #         except NotImplementedError:
    #             clipped_data = out_image[0]
    #         return clipped_data
    
    # vx_full = None
    # vy_full = None
    w_full = None

    if vxfile is not None and vyfile is not None:
        # case = 1
        vx_full = _clip(vxfile, static_area)
        vy_full = _clip(vyfile, static_area)
        if int(nodata) != -9999:
            warnings.warn("NoData values other than -9999 has not implemented yet. Falling back to -9999.")
        nonNaN_pts_idx = np.logical_and(vx_full > -9998, vy_full > -9998)
        vx_full = vx_full[nonNaN_pts_idx]  # remove NaN points
        vy_full = vy_full[nonNaN_pts_idx]  # remove NaN points
        if wfile is not None:
            w_full = _clip(wfile, static_area)
            w_full = w_full[nonNaN_pts_idx]  # remove NaN points
        # return vx_full, vy_full
    # elif vfile is not None:
    #     case = 2
    #     v = clip(vfile, geoms)
    #     v = v[v > -9998]  # remove NaN points
    else:
        # case = 0
        raise TypeError('Vxfile and Vyfile are required.')
  
    # if case == 1:
    if wfile is not None:
        xy_full = np.vstack([vx_full, vy_full, w_full])
    else:
        xy_full = np.vstack([vx_full, vy_full])

    # if len(vx_full) > max_n:
    #     ## See https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy.random.Generator.choice
    #     rng = np.random.default_rng()
    #     xy = rng.choice(xy_full, size=max_n, replace=False, axis=1)
    #     # xy = xy_full
    # else:
    #     xy = xy_full
    
    xy = xy_full
    # xy = xy_full[:, :7000]
        
    # kde2 = KernelDensity(kernel="epanechnikov", bandwidth=2 * 0.6).fit(cc12)
    # log_density2 = kde2.score_samples(cc12)
    # density2 = np.exp(log_density2)
    
    xycov = np.cov(xy)
    # xystd = (xycov[0, 0] * xycov[1, 1]) ** (0.25)
    xystd = (xycov[0, 0] * xycov[1, 1]) ** (0.25)
    #### According to https://doi.org/10.1016/j.spl.2012.07.020 for the epanechnikov kernel
    # bandwidth = 0.4
    bandwidth = 2.1991 * xystd * xy.shape[1] **(-1. / (2 + 4))   # epane
    # bandwidth = 1.0000 * xystd * xy.shape[1] **(-1. / (2 + 4))    # gaussian
    
    midx = np.median(xy[0, :])
    midy = np.median(xy[1, :])
    
    xeval = np.linspace(midx - (thres_sigma + 1) * xystd, midx + (thres_sigma + 1) * xystd, kdegrid_size)
    yeval = np.linspace(midy - (thres_sigma + 1) * xystd, midy + (thres_sigma + 1) * xystd, kdegrid_size)
    xevalg, yevalg = np.meshgrid(xeval, yeval)
    halfd = (xevalg[1] - xevalg[0])/2
    xyeval = np.vstack([xevalg.flatten(), yevalg.flatten()]).T
    
    if wfile is not None:
        pass
        # w = xy[2, :]
        # w = np.where(w < 0, 0, w)
        # kernel = gaussian_kde(xy[:2, :], weights=w)
        # z = kernel(xy[:2, :])
    else:
        kernel = KernelDensity(kernel="epanechnikov", bandwidth=bandwidth).fit(xy.T)
        # kernel = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(xy.T)
        # kernel = KernelDensity(kernel="gaussian", bandwidth=2 * 0.6).fit(xy.T)
        # log_density = kernel.score_samples(xy.T)
        log_density = kernel.score_samples(xyeval)
        z = np.exp(log_density)
        # kernel = gaussian_kde(xy)
        # z = kernel(xy[:, :max_n])
        # z = 1
        zg = np.reshape(z.T, np.shape(xevalg))
        
    vx = xyeval[:, 0]
    vy = xyeval[:, 1]
    # thres_idx = 1
    
    # return vx, vy, z, thres_idx

    thres_multiplier = np.e ** (thres_sigma ** 2 / 2)   # normal dist., +- sigma number 
    thres = max(z) / thres_multiplier
    thres_idx = z >= thres
    idx = thres_idx    # alias
    
    refined_spacing = ((max(vx[thres_idx]) - min(vx[thres_idx])) + (max(vy[thres_idx]) - min(vy[thres_idx]))) / 2

    xeval2 = np.linspace(midx - 2 * refined_spacing, midx + 2 * refined_spacing, kdegrid_size)
    yeval2 = np.linspace(midy - 2 * refined_spacing, midy + 2 * refined_spacing, kdegrid_size)
    xevalg2, yevalg2 = np.meshgrid(xeval2, yeval2)
    halfd2 = (xevalg2[1] - xevalg2[0])/2
    xyeval2 = np.vstack([xevalg2.flatten(), yevalg2.flatten()]).T
    log_density2 = kernel.score_samples(xyeval2)
    z2 = np.exp(log_density2)
    zg2 = np.reshape(z2.T, np.shape(xevalg2))
    thres2 = max(z2) / thres_multiplier
    thres_idx2 = z2 >= thres2
    idx2 = thres_idx2    # alias
    
    vx2 = xyeval2[:, 0]
    vy2 = xyeval2[:, 1]
    
    if plot == 'full':
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # viridis = cm.get_cmap('viridis', 12)
        pt_style = {'s': 6, 'edgecolor': None}
        
        # pt_style = {'s': 1, 'edgecolor': None}
        # plt.scatter(c1, c2, c='gray', alpha=0.5, **pt_style)
        # plt.pcolormesh(xv, yv, density_plot, alpha=0.7, shading='nearest', cmap=cm.oslo_r)
        # plt.axis('equal');

        # if vx_full is not None:
        
        # ax.pcolormesh(xevalg, yevalg, zg, alpha=0.7, shading='nearest', cmap=cramericm.oslo_r)
        ax.scatter(vx_full, vy_full, color='xkcd:gray', alpha=0.1, **pt_style)

        # ax.scatter(vx[idx], vy[idx], c=z[idx], **pt_style)
        # ax.scatter(vx[~idx], vy[~idx], color=viridis(0), alpha=0.4, **pt_style)
    elif plot == 'zoom1':
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pt_style = {'s': 1, 'edgecolor': None}
        ax.scatter(vx_full, vy_full, color='xkcd:pink', alpha=0.6, **pt_style)
        ax.pcolormesh(xevalg2, yevalg2, zg2, alpha=0.85, shading='nearest', cmap=cramericm.oslo_r)
        ax.set_xlim(min(xeval2), max(xeval2))
        ax.set_ylim(min(yeval2), max(yeval2))
        rect = patches.Rectangle((min(vx2[thres_idx2]), min(vy2[thres_idx2])), max(vx2[thres_idx2]) - min(vx2[thres_idx2]), max(vy2[thres_idx2]) - min(vy2[thres_idx2]), 
                         linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.set_title('$e_x$ = {:6.3f} | $e_y$ = {:6.3f} (m/day)'.format(0.5 * (max(vx2[thres_idx2]) - min(vx2[thres_idx2])), 
                                                                 0.5 * (max(vy2[thres_idx2]) - min(vy2[thres_idx2]))))
        
    elif plot == 'zoom2':
        pass
        
    return vx, vy, zg2, thres_idx2, xyeval2

    # if peak_loc:
    #     peak_x = vx[np.argmax(z)]
    #     peak_y = vy[np.argmax(z)]
    #     return vx, vy, z, thres_idx, peak_x, peak_y
    # else:
    #     return vx, vy, z, thres_idx
    
    #     elif case == 2:

    #         if plot:
    #             if ax is None:
    #                 fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    #         bins = ax.hist(v ** 2, 100);
    #         return v, bins

def off_ice_errors(vfile=None, vxfile=None, vyfile=None, wfile=None, off_ice_area=None, thres_sigma=3.0, plot=True, ax=None, max_n=10000, peak_loc=False):
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
            
        if peak_loc:
            peak_x = vx[np.argmax(z)]
            peak_y = vy[np.argmax(z)]
            return vx, vy, z, thres_idx, peak_x, peak_y
        else:
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
    
    
    
def sobel_test2(vxfile=None, vyfile=None):
    '''
    Adopting the concept of strain rate.
    Tensile is defined positive?
    '''
    with rasterio.open(vxfile) as srcx, rasterio.open(vyfile) as srcy:
        vx_full = srcx.read(1)
        vy_full = srcy.read(1)
    
    nonNaN_pts_idx = np.logical_and(vx_full > -9998, vy_full > -9998)
    vx_full[~nonNaN_pts_idx] = np.nan  # replace NaN points with np.nan
    vy_full[~nonNaN_pts_idx] = np.nan  # replace NaN points with np.nan
    # =========== Note: this is to transform from image axis to cartesian axis
    vy_full = np.flipud(vy_full)
    # ===========
        
    # mag_full = np.hypot(vx_full, vy_full)

    exx = sobel(vx_full, axis=1, mode='constant')
    eyy = sobel(vy_full, axis=0, mode='constant')
    duydx = sobel(vy_full, axis=1, mode='constant')
    # =========== transfer back to image axis
    eyy = np.flipud(eyy)
    duydx = np.flipud(duydx)
    # ===========
    
    # =========== Note: this is to correct a reversed Sobel filter along the y direction.
    vx_full = np.flipud(vx_full)
    # ===========
    duxdy = sobel(vx_full, axis=0, mode='constant')
    # =========== transfer back to image axis
    duxdy = np.flipud(duxdy)
    
    exy = 0.5 * (duxdy + duydx)
    
    theta = 0.5 * np.arctan(2 * exy / (exx - eyy))
    e1 = 0.5 * (exx + eyy) + (exy ** 2 + 0.25 * (exx - eyy) ** 2 ) ** 0.5
    e2 = 0.5 * (exx + eyy) - (exy ** 2 + 0.25 * (exx - eyy) ** 2 ) ** 0.5

    # smx = sobel(mag_full,axis=0,mode='constant')
    # smy = sobel(mag_full,axis=1,mode='constant')
    # Get square root of sum of squares
    # sobelm = np.hypot(smx,smy)
    # sobelaz = np.arctan(smy / smx)
    
    return exx, eyy, exy, duxdy, duydx, theta, e1, e2

def sobel_test3(vxfile=None, vyfile=None):
    '''
    Adopting the concept of strain rate.
    Tensile is defined positive?
    Rotated to the along-flow direction.
    '''
    with rasterio.open(vxfile) as srcx, rasterio.open(vyfile) as srcy:
        vx_full = srcx.read(1)
        vy_full = srcy.read(1)
    
    nonNaN_pts_idx = np.logical_and(vx_full > -9998, vy_full > -9998)
    vx_full[~nonNaN_pts_idx] = np.nan  # replace NaN points with np.nan
    vy_full[~nonNaN_pts_idx] = np.nan  # replace NaN points with np.nan
    theta = np.arctan2(vx_full, vy_full)   # along-flow direction (azimuth angle, clockwise from north)
    # =========== Note: this is to transform from image axis to cartesian axis
    vy_full = np.flipud(vy_full)
    # ===========
        
    # mag_full = np.hypot(vx_full, vy_full)

    exx = sobel(vx_full, axis=1, mode='constant')
    eyy = sobel(vy_full, axis=0, mode='constant')
    duydx = sobel(vy_full, axis=1, mode='constant')
    # =========== transfer back to image axis
    eyy = np.flipud(eyy)
    duydx = np.flipud(duydx)
    # ===========
    
    # =========== Note: this is to correct a reversed Sobel filter along the y direction.
    vx_full = np.flipud(vx_full)
    # ===========
    duxdy = sobel(vx_full, axis=0, mode='constant')
    # =========== transfer back to image axis
    duxdy = np.flipud(duxdy)
    
    exy = 0.5 * (duxdy + duydx)
    
    exx_rot = exx * np.cos(theta) ** 2 + eyy * np.sin(theta) **2 + exy * np.sin(2 * theta)
    eyy_rot = exx * np.sin(theta) ** 2 + eyy * np.cos(theta) **2 - exy * np.sin(2 * theta)
    exy_rot = 0.5 * (eyy - exx) * np.sin(2 * theta) +  exy * np.cos(2 * theta)
    
    return exx, eyy, exy, duxdy, duydx, theta, exx_rot, eyy_rot, exy_rot


def sobel_strain_test(vxfile=None, vyfile=None, wfile=None, on_ice_area=None, thres_sigma=3.0, plot=True, ax=None, max_n=10000, max_s=100, return_sobelimage=False):
    """

    """ 
    shapefile = gpd.read_file(on_ice_area)
    geoms = shapefile.geometry.values
    geoms = [mapping(geoms[i]) for i in range(len(geoms))]
    with rasterio.open(vxfile) as srcx, rasterio.open(vyfile) as srcy:
        vx_full = srcx.read(1)
        vy_full = srcy.read(1)
        transform = srcx.transform
        dx = transform[0]
        dy = abs(transform[4])
    if wfile is not None:
        with rasterio.open(wfile) as srcw:
            w_full = srcw.read(1)
    else:
        w_full = None
        
    nonNaN_pts_idx = np.logical_and(vx_full > -9998, vy_full > -9998)
    vx_full[~nonNaN_pts_idx] = np.nan  # replace NaN points with np.nan
    vy_full[~nonNaN_pts_idx] = np.nan  # replace NaN points with np.nan
    flow_theta = np.arctan2(vx_full, vy_full)   # along-flow direction (azimuth angle, clockwise from north)
    # =========== Note: this is to transform from image axis to cartesian axis
    vy_full = np.flipud(vy_full)
    # ===========
        
    # mag_full = np.hypot(vx_full, vy_full)

    exx = sobel(vx_full, axis=1, mode='constant')
    eyy = sobel(vy_full, axis=0, mode='constant')
    duydx = sobel(vy_full, axis=1, mode='constant')
    # =========== transfer back to image axis
    eyy = np.flipud(eyy)
    duydx = np.flipud(duydx)
    # ===========
    
    # =========== Note: this is to correct a reversed Sobel filter along the y direction.
    vx_full = np.flipud(vx_full)
    # ===========
    duxdy = sobel(vx_full, axis=0, mode='constant')
    # =========== transfer back to image axis
    duxdy = np.flipud(duxdy)
    
    # exx /= dx
    # eyy /= dy
    # duxdy /= dy
    # duydx /= dx
    
    exy = 0.5 * (duxdy + duydx)
    
    exx_rot = exx * np.cos(flow_theta) ** 2 + eyy * np.sin(flow_theta) **2 + exy * np.sin(2 * flow_theta)
    eyy_rot = exx * np.sin(flow_theta) ** 2 + eyy * np.cos(flow_theta) **2 - exy * np.sin(2 * flow_theta)
    exy_rot = 0.5 * (eyy - exx) * np.sin(2 * flow_theta) +  exy * np.cos(2 * flow_theta)
    
    theta = 0.5 * np.arctan(2 * exy / (exx - eyy))
    e1 = 0.5 * (exx + eyy) + (exy ** 2 + 0.25 * (exx - eyy) ** 2 ) ** 0.5
    e2 = 0.5 * (exx + eyy) - (exy ** 2 + 0.25 * (exx - eyy) ** 2 ) ** 0.5
    
    e1_masked = mask_by_shp(shapefile['geometry'], e1, rasterio.open(vxfile))
    e2_masked = mask_by_shp(shapefile['geometry'], e2, rasterio.open(vxfile))
    exy_rot_masked = mask_by_shp(shapefile['geometry'], exy_rot, rasterio.open(vxfile))
    
    e_NaN_pts_idx     = np.isnan(e1_masked)
    # sobel_outlier_pts_idx = np.logical_or(sx_full > max_s, sy_full > max_s)
    # sobel_bad_pts_idx = np.logical_or(sobel_NaN_pts_idx, sobel_outlier_pts_idx)
    e1_masked = e1_masked[~e_NaN_pts_idx]
    e2_masked = e2_masked[~e_NaN_pts_idx]
    exy_rot_masked = exy_rot_masked[~e_NaN_pts_idx]
    
    # if w_full is not None:
    #     w_full = w_full[~sobel_bad_pts_idx]
    #     xy_full = np.vstack([sx_full, sy_full, w_full])
    # else:
    #     xy_full = np.vstack([sx_full, sy_full])
    
    e_full = np.vstack([e1_masked, e2_masked])
    
    if len(e1_masked) > max_n:
        ## See https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy.random.Generator.choice
        rng = np.random.default_rng()
        e = rng.choice(e_full, size=max_n, replace=False, axis=1)
    else:
        e = e_full
    
    # if wfile is not None:
    #     w = xy[2, :]
    #     w = np.where(w < 0, 0, w)
    #     kernel = gaussian_kde(xy[:2, :], weights=w)
    #     z = kernel(xy[:2, :])
    # else:
    kernel = gaussian_kde(e)
    z = kernel(e)
    
    e1s = e[0, :]
    e2s = e[1, :]
            
    thres_multiplier = np.e ** (thres_sigma ** 2 / 2)   # normal dist., +- sigma number 
    thres = max(z) / thres_multiplier
    thres_idx = z >= thres
    idx = thres_idx    # alias
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        viridis = cm.get_cmap('viridis', 12)
        pt_style = {'s': 6, 'edgecolor': None}

        if e1_masked is not None:
            ax.scatter(e1_masked, e2_masked, color='xkcd:gray', alpha=0.2, **pt_style)

        ax.scatter(e1s[idx], e2s[idx], c=z[idx], **pt_style)
        ax.scatter(e1s[~idx], e2s[~idx], color=viridis(0), alpha=0.4, **pt_style)
        
    if return_sobelimage:
        return e1s, e2s, z, thres_idx, exx, eyy, exy, duxdy, duydx, theta, e1, e2, exy_rot, exy_rot_masked, exx_rot, eyy_rot
    else:
        return e1s, e2s, z, thres_idx
    

    
def sobel_scattering(vxfile=None, vyfile=None, wfile=None, on_ice_area=None, thres_sigma=3.0, plot=True, ax=None, max_n=10000, max_s=100, return_sobelimage=False):
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
        
    if return_sobelimage:
        return sx, sy, z, thres_idx, sobelx, sobely
    else:
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
