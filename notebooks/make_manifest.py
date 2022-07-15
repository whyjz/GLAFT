#!/usr/bin/env python

### This script creates `manifest.csv`, the necessary metadata of the files to be used in study.

import pandas as pd
import glob

def create_row(date: str, duration: int, chip: str, 
               pixel_spacing: int, res: str, prefilter: str, 
               subpixel: str, software: str, i: str, j: str):
    row = {'Date':               date,      
           'Duration (days)':    duration,
           'Template size (px)': chip,
           'Template size (m)':  int(chip) * pixel_spacing,
           'Pixel spacing (px)': res,
           'Pixel spacing (m)':  int(res) * pixel_spacing,
           'Prefilter':          prefilter, 
           'Subpixel':           subpixel, 
           'Software':           software,
           'Vx':                 i,                
           'Vy':                 j,
          }
    return row

def fill_satspecs_into_kargs(kargs, case=None):
    if case == 1:
        kargs['date'] = 'LS8-20180304-20180405'
        kargs['duration'] = 32
        kargs['pixel_spacing'] = 15
    elif case == 2:
        kargs['date'] = 'LS8-20180802-20180818'
        kargs['duration'] = 16
        kargs['pixel_spacing'] = 15
    elif case == 3:
        kargs['date'] = 'Sen2-20180304-20180314'
        kargs['duration'] = 10
        kargs['pixel_spacing'] = 10
    elif case == 4:
        kargs['date'] = 'Sen2-20180508-20180627'
        kargs['duration'] = 50
        kargs['pixel_spacing'] = 10
    return kargs

df = pd.DataFrame(columns = ['Date', 'Duration (days)', 'Template size (px)', 'Template size (m)', 'Pixel spacing (px)', 'Pixel spacing (m)', 'Prefilter', 'Subpixel', 'Software', 'Vx', 'Vy'])

carst_vxs = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/CARST/*/*velo-raw_vx.tif'))
carst_vys = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/CARST/*/*velo-raw_vy.tif'))





## parsing CARST products
for i, j in zip(carst_vxs, carst_vys):
    kargs = {}
    kargs['i'] = i
    kargs['j'] = j
    kargs['software'] = 'CARST'
    info = i.split('/')[-2]
    date_raw =  info.split('_')[1]
    kargs['chip'] =      info.split('_')[3][-2:]
    kargs['res'] =       info.split('_')[4][4:]
    kargs['prefilter'] = info.split('_')[5]
    kargs['subpixel'] = '16-node oversampling'
    if date_raw == '20180314':
        kargs = fill_satspecs_into_kargs(kargs, case=3)
    elif date_raw == '20180405':
        kargs = fill_satspecs_into_kargs(kargs, case=1)
    elif date_raw == '20180627':
        kargs = fill_satspecs_into_kargs(kargs, case=4)
    elif date_raw == '20180818':
        kargs = fill_satspecs_into_kargs(kargs, case=2)
    else:
        raise ValueError('Strange date string!')
    row = create_row(**kargs)
    df = pd.concat([df, pd.DataFrame.from_records(row, index=[0])], ignore_index=True)

    
    
    
    
## parsing GIV products
giv_vxs = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/GIV/u*.tif'))
giv_vys = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/GIV/v*.tif'))

for i, j in zip(giv_vxs, giv_vys):
    kargs = {}
    kargs['i'] = i
    kargs['j'] = j
    kargs['software'] = 'GIV'
    info = i.split('/')[-1]
    date_raw =  info.split('_')[1]
    # chip =      'multi'
    res_raw =   info.split('_')[3][:-4]
    prefilter_raw = info.split('_')[2]
    kargs['subpixel'] = 'interest point groups'
    if date_raw == 's12':
        kargs = fill_satspecs_into_kargs(kargs, case=3)
    elif date_raw == 'l12':
        kargs = fill_satspecs_into_kargs(kargs, case=1)
    elif date_raw == 's34':
        kargs = fill_satspecs_into_kargs(kargs, case=4)
    elif date_raw == 'l34':
        kargs = fill_satspecs_into_kargs(kargs, case=2)
    else:
        raise ValueError('Strange date string!')
    if res_raw == '50':
        if date_raw == 's12' or date_raw == 's34':
            res_px = '4.003'
            res_m = '40.03'
        elif date_raw == 'l12' or date_raw == 'l34':
            res_px = '4.009'
            res_m = '60.14'
        else:
            raise ValueError('Something wrong! #1')
    elif res_raw == '200':
        if date_raw == 's12' or date_raw == 's34':
            res_px = '16.04'
            res_m = '160.4'
        elif date_raw == 'l12' or date_raw == 'l34':
            res_px = '15.13'
            res_m = '242.1'
        else:
            raise ValueError('Something wrong! #2')
    else:
        raise ValueError('Strange resolution!')
    if prefilter_raw == 'gaus':
        kargs['prefilter'] = 'Gau'
    elif prefilter_raw == 'NAOF':
        kargs['prefilter'] = 'NAOF'
    elif prefilter_raw == 'r':
        kargs['prefilter'] = 'None'
    else:
        raise ValueError('Strange prefilter!')
    row = {'Date':               kargs['date'],      
           'Duration (days)':    kargs['duration'],
           'Template size (px)': 'varying: multi-pass',
           'Template size (m)':  'varying: multi-pass',
           'Pixel spacing (px)': res_px,
           'Pixel spacing (m)':  res_m,
           'Prefilter':          kargs['prefilter'], 
           'Subpixel':           kargs['subpixel'], 
           'Software':           kargs['software'],
           'Vx':                 kargs['i'],                
           'Vy':                 kargs['j'],
          }
    df = pd.concat([df, pd.DataFrame.from_records(row, index=[0])], ignore_index=True)
    
    
    
    
    
## parsing Vmap products
vmap_vxs = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/Vmap/**/*vx.tif', recursive=True))
vmap_vys = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/Vmap/**/*vy.tif', recursive=True))

for i, j in zip(vmap_vxs, vmap_vys):
    kargs = {}
    kargs['i'] = i
    kargs['j'] = j
    kargs['software'] = 'Vmap'
    info = i.split('/')[-1]
    kargs['res'] = '1'
    if 'parabolic_subpixel' in i:
        kargs['chip'] = info.split('_')[-3][:2]
        if 'gaussfilter' in i:
            kargs['prefilter'] = 'Gau'
            kargs['subpixel'] = 'parabolic'
        elif 'noprefilter' in i:
            kargs['prefilter'] = 'None'
            kargs['subpixel'] = 'parabolic'
        else:
            raise ValueError('Something wrong! #3')
    elif 'subpixel_comparison' in i:
        kargs['prefilter'] = 'LoG'
        kargs['chip'] = '31'
        if 'spm1' in info:
            kargs['subpixel'] = 'parabolic'
        elif 'spm2' in info:
            kargs['subpixel'] = 'affine adaptive'
        elif 'spm3' in info:
            kargs['subpixel'] = 'affine'
        else:
            raise ValueError('Something wrong! #5')
    else:
        raise ValueError('Something wrong! #4')
    if info.startswith('L'):
        if '20180304' in info:
            kargs = fill_satspecs_into_kargs(kargs, case=1)
        elif '20180802' in info:
            kargs = fill_satspecs_into_kargs(kargs, case=2)
        else:
            raise ValueError('Something wrong! #6')
    elif info.startswith('T'):
        if '20180304' in info:
            kargs = fill_satspecs_into_kargs(kargs, case=3)
        elif '20180508' in info:
            kargs = fill_satspecs_into_kargs(kargs, case=4)
        else:
            raise ValueError('Something wrong! #7')
    else:
        raise ValueError('Something wrong! #8')
        
    row = create_row(**kargs)
    df = pd.concat([df, pd.DataFrame.from_records(row, index=[0])], ignore_index=True)

    
    
    
    
## parsing autoRIFT products
autorift_vxs = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/autoRIFT/**/*vx.tif', recursive=True))
autorift_vys = sorted(glob.glob('/home/jovyan/Projects/PX_comparison/PX/autoRIFT/**/*vy.tif', recursive=True))

for i, j in zip(autorift_vxs, autorift_vys):
    kargs = {}
    kargs['i'] = i
    kargs['j'] = j
    kargs['software'] = 'autoRIFT'
    info = i.split('/')[-2]
    kargs['subpixel'] = 'pyrUP'
    kargs['res'] = info.split('_')[-1]
    kargs['chip'] = info.split('_')[-4][:2]
    if 'gauss_hp' in i:
        kargs['prefilter'] = 'Gau'
    elif 'naof2' in i:
        kargs['prefilter'] = 'NAOF'
    elif 's__' in i:
        kargs['prefilter'] = 'None'
    else:
        raise ValueError('Strange prefilter string!')
    if 'Landsat' in i:
        if '20180304' in info:
            kargs = fill_satspecs_into_kargs(kargs, case=1)
        elif '20180802' in info:
            kargs = fill_satspecs_into_kargs(kargs, case=2)
        else:
            raise ValueError('Strange date string! (autoRIFT) #1')
    elif 'S2' in i:
        if '20180304' in info:
            kargs = fill_satspecs_into_kargs(kargs, case=3)
        elif '20180508' in info:
            kargs = fill_satspecs_into_kargs(kargs, case=4)
        else:
            raise ValueError('Strange date string! (autoRIFT) #2')
    else:
        raise ValueError('Strange date string! (autoRIFT) #3')
    row = create_row(**kargs)
    df = pd.concat([df, pd.DataFrame.from_records(row, index=[0])], ignore_index=True)
    
    
df.to_csv('manifest.csv', index=False)