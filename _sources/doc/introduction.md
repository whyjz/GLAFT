# GLAFT User Manual

GLAcier Feature Tracking testkit (GLAFT) is based on the scientific Python ecosystem and focuses on calculating the metrics and benchmarking glacier velocity products derived using the feature tracking technique. To be compatible with most feature tracking tools, GLAFT aims to process only the most common and essential product, which is the velocity itself (with an optional input of reliability file used as weight for calculating KDE). GLAFT is also equipped with visualization tools for users to perform qualitative assessments data and discover hidden patterns in the data. 

GLAFT is an open source project with all source code hosted on Github (https://github.com/whyjz/GLAFT). Users can find relevant documentation and cloud-executable demos in the same repository and on its Jupyter Book-based Github pages (https://whyjz.github.io/GLAFT/). GLAFT is also available through pip installation as well as Ghub.

## Input

The current version of GLAFT accepts velocity data with x and y components stored in separate raster files (with a format compatible with the Python rasterio package, such as GeoTiff). 

## Functions

- Data and metadata input 
- Functions for calculating metrics
- Auxiliary functions
- Visualization tools
- Details about finding the KDE
- Two-step gridding
