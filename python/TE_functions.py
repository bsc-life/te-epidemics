#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Authors: Miguel Ponce de Leon, Camila F. T. Pontes

__author__ = "Miguel Ponce de León"
__email__  = "miguel.ponce@bsc.es"

import numpy as np
import xarray as xr

def calculate_directionality_index(Txy):

    DIxy = Txy.values - Txy.values.transpose(1, 0, 2)
    DIxy = xr.DataArray(DIxy, dims=Txy.dims, coords=Txy.coords)
    
    return DIxy

