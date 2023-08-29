# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy as sp
import tqdm
import os

os.chdir(r"C:\Users\speed\Documents\#Stellar Classification")

def density_minmax(x, thresh = 0.9):
    q = np.quantile(x, q = thresh)
    x_std = (x - x.min(axis=0)) / (q - x.min(axis=0))
    return x_std

# %% data

hdul = fits.open('MaStar\mastar-combspec-v3_1_1-v1_7_7-lsfpercent99.5.fits.gz')
cata = hdul[1].data
cols = hdul[1].columns
hdul.close()

sdss_spectra = np.load('sdss_ex.npy')
cflib_spectra = np.load('cflib.npy')

hdul = fits.open('CFLIB/BINTABLE/249.fits')

data = hdul[1].data
hdu = hdul[1].header

hdul.close()

# %%
x1 = sdss_spectra['WAVE']
x2 = data['wavelength'][0]

x1_limit = (x1 < x2[-1])
x2_limit = (x2 > x1[0])

x1_l, x2_l = x1[x1_limit], x2[x2_limit]

#%% removing data with too much nan and interpolating the rest

count_zero = lambda x: np.count_nonzero(np.isclose(x,0,atol=1e-4))
zeros = np.apply_along_axis(count_zero, 1, cflib_spectra['FLUX'])

idx = zeros<125
useful = cflib_spectra['FLUX'][idx]

nan = np.where(np.isclose(useful, 0, atol=1e-4), np.nan, useful)
#xs = np.tile(x2,(844,1))

#a = sp.interpolate.interp1d(xs, nan, kind='linear', axis=1)

inter = pd.DataFrame(nan).interpolate(method='linear', axis=1).values

cflib_nonan = np.ndarray((844,), dtype=[('FLUX', '<f8', 15011), ('NAME', '<U10')])

cflib_nonan['FLUX'] = inter
cflib_nonan['NAME'] = cflib_spectra['NAME'][idx]

# %% spline interpolation to fixed grid

def linear(x):
    return np.linspace(x[0], x[-1], len(x))

sx = linear(x1_l)

def cflib_spline(data):
    data_l = data[x2_limit]

    xa = sp.interpolate.CubicSpline(x2_l, data_l)
    
    return xa(sx)

cfl = np.apply_along_axis(cflib_spline, 1, cflib_nonan['FLUX'])

arr = np.ndarray((844,), dtype=[('FLUX', '<f8', 4175), ('NAME', 'U22')])

arr['FLUX'] = cfl
arr['NAME'] = cflib_spectra['NAME'][idx]


# %%

np.save('processed_cflib2.npy', arr)

# %% unlabeled preprocess
def sdss_spline_inf(data):
    data_l = data[x1_limit]
    np.put(data_l, np.where(data_l == np.inf)[0], 0)
  
    return data_l

sdss_unlabeled = np.apply_along_axis(sdss_spline_inf, 1, cata['FLUX'])

# %%

zeros2 = np.apply_along_axis(count_zero, 1, sdss_unlabeled)
 
nan = np.where(np.isclose(sdss_unlabeled, 0, atol=1e-4), np.nan, sdss_unlabeled)

inter = pd.DataFrame(nan).interpolate(method='linear', axis=1).values

def rep_na_at_start(row):

    nans = np.isnan(row)
    if sum(nans) != 0:
        ix = np.where(np.diff(nans.astype('int'))==-1)[0][0]
        row[nans] = row[ix+1]
    return row

inter = np.apply_along_axis(rep_na_at_start, 1, inter)
#sdss_nonan = np.ndarray((23893,), dtype=[('FLUX', '<f8', 4175)])
 
#sdss_nonan['FLUX'] = inter

# %%
np.save('processed_sdss_unlabeled2.npy', inter)
