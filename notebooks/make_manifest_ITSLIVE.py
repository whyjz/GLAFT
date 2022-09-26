#!/usr/bin/env python

### This script creates `manifest_ITSLIVE.csv`, the necessary metadata of the files to be used in study.

import pandas as pd
import glob
from datetime import datetime

def create_row(date: str, duration: int, chip: str, 
               res: str, prefilter: str, 
               subpixel: str, software: str, i: str, j: str,
               startdate: str, enddate: str):
    row = {'Label':              date,
           'Start date':         startdate,
           'End date':           enddate,
           'Duration (days)':    duration,
           'Template size (px)': 'NaN',
           'Template size (m)':  chip,
           'Pixel spacing (px)': 'NaN',
           'Pixel spacing (m)':  res,
           'Prefilter':          prefilter, 
           'Subpixel':           subpixel, 
           'Software':           software,
           'Vx':                 i,                
           'Vy':                 j,
          }
    return row



df = pd.DataFrame(columns = ['Label', 'Start date', 'End date', 'Duration (days)', 'Template size (px)', 'Template size (m)', 'Pixel spacing (px)', 'Pixel spacing (m)', 'Prefilter', 'Subpixel', 'Software', 'Vx', 'Vy'])

itslive_vxs = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/ITS_LIVE/*vx.tif'))
itslive_vys = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/ITS_LIVE/*vy.tif'))

for i, j in zip(itslive_vxs, itslive_vys):
    kargs = {}
    kargs['i'] = i
    kargs['j'] = j
    kargs['software'] = 'ITS_LIVE (autoRIFT)'
    kargs['chip'] =      'varying (240-480)'
    kargs['res'] =       120
    kargs['prefilter'] = 'ITS_LIVE (5x5 Wallis operator)' 
    kargs['subpixel'] = 'ITS_LIVE (16-node oversampling)'
    basename = i.split('/')[-1]
    if basename.startswith('LC08'):
        startdate = basename.split('_')[3]
        enddate = basename.split('_')[11]
        kargs['date'] = 'LS8-{}-{}'.format(startdate, enddate)
    elif basename.startswith('S2'):
        startdate = basename.split('_')[2]
        enddate = basename.split('_')[8]
        kargs['date'] = 'Sen2-{}-{}'.format(startdate, enddate)
    else:
        raise ValueError('Filename parsing error!')
    kargs['startdate'] = startdate
    kargs['enddate'] = enddate
    timedel = datetime.strptime(enddate, '%Y%m%d') - datetime.strptime(startdate, '%Y%m%d')
    kargs['duration'] = timedel.days
    
    row = create_row(**kargs)
    df = pd.concat([df, pd.DataFrame.from_records(row, index=[0])], ignore_index=True)
    
    
    

df.to_csv('manifest_ITSLIVE.csv', index=False)