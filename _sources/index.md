# Gftt: an open-source tool for evaluating remotely sensed glacier velocity products

Whyjay Zheng<sup>1</sup>, Shashank Bhushan<sup>2</sup>, Maximillian S Van Wyk de Vries<sup>3</sup>, William Hardy Kochtitzky<sup>4</sup>, and David E Shean<sup>2</sup>

<sup>1</sup>University of California, Berkeley, Statistics, Berkeley, CA, United States

<sup>2</sup>University of Washington, Civil & Environmental Engineering, Seattle, WA, United States

<sup>3</sup>University of Minnesota Twin Cities, Department of Earth and Environmental Sciences and Saint Anthony Falls Laboratory, Minneapolis, United States

<sup>4</sup>University of Ottawa, Department of Geography, Environment and Geomatics, Ottawa, Canada

Corresponding author and email: Whyjay Zheng (whyjz@berkeley.edu)

## Abstract

Glacier velocity reflects the dynamics of ice flow, and its change over time serves a key role in predicting the future sea-level rise. Glacier feature tracking (also known as offset tracking or pixel tracking) is one of the most widely-used approaches for mapping glacier velocity using remote sensing data. However, running this workflow relies on multiple empirical parameter choices such as correlation kernel selection, image filter, and template size. As each target glacier area has different data availability, surface feature density, and ice flow width, there is no one-size-fits-all parameter set for glacier feature tracking.

Finding an ideal parameter set for a given glacier requires quantitative and objective metrics to determine the quality of resulting velocity maps. The objective of our Glacier feature tracking test (gftt) project is both to devise a set of widely applicable metrics and to build a Python-based tool for calculating them. These metrics can be thus used for comparing the performance of different tracking parameters. We use Kaskawulsh glacier, Canada, as a test case to compare the velocity mapping results using Landsat 8 and Sentinel-2 images, various software packages (including Auto-RIFT, CARST, GIV, and vmap), and a range of input parameters. To begin with, we calculate random error over stable terrain, a metric that has been used for evaluating the uncertainty of the velocity products. We develop two other workflows for exploring new metrics and validating existing metrics, including the test with synthetic pixel offsets and the comparison with GNSS records. These existing and new metrics, calculated through the gftt software, will help determine optimal parameter sets for feature tracking of Kaskawulsh glacier and any other glacier around the world.

This work is supported by the NSF Earth Cube Program under awards 1928406, 1928374.

## Plain language summary

There are many ways to measure how quickly glaciers move including using GPS and satellites. Here, we compare a variety of methods to measure glacier motion to determine which method(s) perform the best in mapping glacier movement. We develop a software package called “gftt” to help execute our work and share our methods with the scientific community.

## How to cite

doi:10.1002/essoar.10509355.1