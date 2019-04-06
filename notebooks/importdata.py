import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table


def importdata():
    """Read in the data from decals and mgc catalogs and do manipulations on them to get them to a usable form.

    In particular, grab the chi^2 values for exponential and de Vocoleur models of the galaxies.

    Use these according to Rongpu's definition to get a classifier that is a probability of being an elliptical or a disk. 
    Set this to 1/2 everywhere where these values cannot be used.

    Also determine the axis ratio.

    Determine the colors.

    Merge these with the relevant information from the mgc catalog that we are trying to fit. 

    Return this catalog."""

    decals_loc = '../downloads/decals-dr7.1-UKBOSS_best_ukwide_v5_2-02jun2015-match.fits'
    mgc_loc = '../downloads/UKBOSS_best_ukwide_v5_2-02jun2015-match.fits'

    # DECaLS catalog
    decals = Table.read(decals_loc)
    # matched best_ukwide catalog
    mgc = Table.read(mgc_loc)

    decals["DCHISQ_EXP"] = decals["DCHISQ"][:,2]
    decals["DCHISQ_DEV"] = decals["DCHISQ"][:,3]

    # calculate parameters for the axis ratio and the probability of being a disk or elliptical
    e = np.zeros(len(decals)) # the e parameter is zero for circularly symmetric profiles (PSF and SIMP)
    mask = (decals['TYPE']=='EXP') | (decals['TYPE']=='EXP ')
    e[mask] = (np.sqrt(decals['SHAPEEXP_E1']**2+decals['SHAPEEXP_E2']**2))[mask]
    mask = (decals['TYPE']=='DEV') | (decals['TYPE']=='DEV ')
    e[mask] = (np.sqrt(decals['SHAPEDEV_E1']**2+decals['SHAPEDEV_E2']**2))[mask]
    mask = (decals['TYPE']=='COMP')
    e[mask] = ((1-decals['FRACDEV']) * np.sqrt(decals['SHAPEEXP_E1']**2+decals['SHAPEEXP_E2']**2) \
              + decals['FRACDEV'] * np.sqrt(decals['SHAPEDEV_E1']**2+decals['SHAPEDEV_E2']**2))[mask]
    q = (1-e)/(1+e)

    # shape probability (definition of shape probability in Soo et al. 2017)
    # this parameter characterizes how well an object is fit by exponential profile vs de Vaucouleurs profile
    p = np.ones(len(decals))*0.5
    # DCHISQ[:, 2] is DCHISQ_EXP; DCHISQ[:, 3] is DCHISQ_DEV
    mask_chisq = (decals['DCHISQ_DEV']>0) & (decals['DCHISQ_EXP']>0)
    p[mask_chisq] = decals['DCHISQ_DEV'][mask_chisq]/(decals['DCHISQ_DEV']+decals['DCHISQ_EXP'])[mask_chisq]

    decals['axis_ratio'] = q
    decals['p_exp'] = p

    del decals["DCHISQ"]

    decals['FLUX_G'] = decals['FLUX_G']/decals['MW_TRANSMISSION_G']
    decals['FLUX_R'] = decals['FLUX_R']/decals['MW_TRANSMISSION_R']
    decals['FLUX_Z'] = decals['FLUX_Z']/decals['MW_TRANSMISSION_Z']
    decals['FLUX_W1'] = decals['FLUX_W1']/decals['MW_TRANSMISSION_W1']
    decals['FLUX_W2'] = decals['FLUX_W2']/decals['MW_TRANSMISSION_W2']

    # Compute extinction-corrected magnitudes and errors for DECaLS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decals['gmag'] = 22.5 - 2.5*np.log10(decals['FLUX_G'])
        decals['rmag'] = 22.5 - 2.5*np.log10(decals['FLUX_R'])
        decals['zmag'] = 22.5 - 2.5*np.log10(decals['FLUX_Z'])
        decals['w1mag'] = 22.5 - 2.5*np.log10(decals['FLUX_W1'])
        decals['w2mag'] = 22.5 - 2.5*np.log10(decals['FLUX_W2'])
        decals['gmagerr'] = 1/np.sqrt(decals['FLUX_IVAR_G'])/decals['FLUX_G']
        decals['rmagerr'] = 1/np.sqrt(decals['FLUX_IVAR_R'])/decals['FLUX_R']
        decals['zmagerr'] = 1/np.sqrt(decals['FLUX_IVAR_Z'])/decals['FLUX_Z']
        decals['w1magerr'] = 1/np.sqrt(decals['FLUX_IVAR_W1'])/decals['FLUX_W1']
        decals['w2magerr'] = 1/np.sqrt(decals['FLUX_IVAR_W2'])/decals['FLUX_W2']

    # Restrict to DECaLS objects with 2+ exposures in grz bands
    mask = (decals['NOBS_G'] >= 2) & (decals['NOBS_R'] >= 2) & (decals['NOBS_Z'] >= 2)
    decals = decals[mask]
    mgc = mgc[mask]

    # Require valid grzW1W2 photometry
    mask = np.isfinite(decals['gmag']) & np.isfinite(decals['rmag']) & np.isfinite(decals['zmag']) & \
            np.isfinite(decals['w1mag']) & np.isfinite(decals['w2mag'])
    decals = decals[mask]
    mgc = mgc[mask]

    # Calculate g-r, r-z, z-w1, w1-w2 color
    decals['gminr'] = decals['gmag']-decals['rmag']
    decals['rminz'] = decals['rmag']-decals['zmag']
    decals['zminw1'] = decals['zmag']-decals['w1mag']
    decals['w1minw2'] = decals['w1mag']-decals['w2mag']

    absmagbest = mgc['ABSMAG_BEST']
    del mgc["ABSMAG_BEST"]

    catalog = decals.copy()

    catalog['redshift'] = mgc['ZBEST']
    catalog['redshift_err'] = mgc['SIGMAZ_BEST']
    catalog['mass_ir'] = mgc['MASS_IR_BEST']
    catalog['mass_ir_err'] = mgc['MASSERR_IR_BEST']
    catalog['mass_opt'] = mgc['MASS_OPT_BEST']
    catalog['mass_opt_err'] = mgc['MASSERR_OPT_BEST']
    catalog['b1000'] = mgc['B1000_IR_BEST']
    catalog['b300'] = mgc['B300_IR_BEST']

    # delete the unimportant columns
    del catalog["BRICKID"]
    del catalog["BRICKNAME"]

    catalog = catalog.to_pandas()

    return catalog
