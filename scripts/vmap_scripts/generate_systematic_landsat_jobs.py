#! /usr/bin/env python
#Picked up from dshean's vmap repository, modified for this project to get pairs with 16 days and more
"""
Given two input scenes, create job scripts for systematic mapping using permutations of all options
"""
#readarray -t LINES <"good_list.txt"
#for LINE in "${LINES[@]}";do mv $LINE ./vmap/; done

import sys,os
from datetime import datetime, timedelta
import numpy as np
from pygeotools.lib import timelib
from itertools import combinations



fn_list = sys.argv[1:]
print(fn_list)
# fn_list contains the 2 images
# iterate_between refinement 1,2,3
# iterate between kernel 21 to 31 with increments of 2 
# produce the above mentioned for loops in reverse
fixed_vmap_opt = ['-dt', 'day', '-filter','-threads','4']

pref = os.path.splitext(os.path.basename(fn_list[0]))[0]+'__'+os.path.splitext(os.path.basename(fn_list[1]))[0]+'_'
outfn = os.path.join(os.path.dirname(fn_list[0]),pref+'dedicated_vmap_jobs.txt')

with open(outfn,'w') as foo :
    for refine_mode in [1,2,3]:
        for kernel_size in np.linspace(21,35,5,dtype=int):
                vmap_args = fn_list
                var_vmap_opt = ['-refinement',str(refine_mode),'-kernel',str(kernel_size)]
                vmap_cmd = ['vmap.py']+fixed_vmap_opt+var_vmap_opt+vmap_args
                vmap_str = ' '.join(vmap_cmd)+'\n'
                foo.write(vmap_str)
print("Script is complete")

