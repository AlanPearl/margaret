import os, warnings
import numpy as np
from astropy.table import Table

MARGPATH = os.path.abspath(os.path.join(__file__, "..", ".."))
DOWNLOADS = os.path.join(MARGPATH, "downloads")


def importdata(data_slice=slice(None), mask_stars=True, mask_rad30=False, mask_chisq0=False):
    """Read in the data from decals and mgc catalogs and do manipulations on them to get them to a usable form.

    In particular, grab the chi^2 values for exponential and de Vocoleur models of the galaxies.

    Use these according to Rongpu's definition to get a classifier that is a probability of being an elliptical or a disk. 
    Set this to 1/2 everywhere where these values cannot be used.

    Also determine the axis ratio.

    Determine the colors.

    Merge these with the relevant information from the mgc catalog that we are trying to fit. 

    Return this catalog."""

    decals_filename = 'decals-dr7.1-UKBOSS_best_ukwide_v5_2-02jun2015-match.fits'
    mgc_filename = 'UKBOSS_best_ukwide_v5_2-02jun2015-match.fits'
    
    decals_loc = os.path.join(DOWNLOADS, decals_filename)
    mgc_loc = os.path.join(DOWNLOADS, mgc_filename)

    # DECaLS catalog
    decals = Table.read(decals_loc)[data_slice]
    # matched best_ukwide catalog
    mgc = Table.read(mgc_loc)[data_slice]

    decals["DCHISQ_EXP"] = decals["DCHISQ"][:,2]
    decals["DCHISQ_DEV"] = decals["DCHISQ"][:,3]
    
    absmagbest = mgc['ABSMAG_BEST']
    del mgc["ABSMAG_BEST"]
    del decals["DCHISQ"]
    
    decals = decals.to_pandas()
    mgc = mgc.to_pandas()

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
    
    prob_exp = np.full(len(decals), 0.5)
    logprob_exp = logprob_dev = np.full(len(decals), np.log(0.5))
    expscale = np.zeros(len(decals))
    
    log_sum = np.logaddexp(-0.5 * decals["DCHISQ_EXP"][mask_chisq],
                           -0.5 * decals["DCHISQ_DEV"][mask_chisq])
    logprob_exp[mask_chisq] = -0.5 * decals["DCHISQ_EXP"][mask_chisq] - log_sum
    logprob_dev[mask_chisq] = -0.5 * decals["DCHISQ_DEV"][mask_chisq] - log_sum
    
    prob_exp[mask_chisq] = np.exp(logprob_exp[mask_chisq])
    expscale[mask_chisq] = logprob_exp[mask_chisq] - logprob_dev[mask_chisq]

    decals['axis_ratio'] = q
    decals['p_exp'] = p
    decals['prob_exp'] = prob_exp
    decals['expscale'] = expscale


    # Compute extinction-corrected magnitudes and errors for DECaLS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        decals['FLUX_G'] /= decals['MW_TRANSMISSION_G']
        decals['FLUX_R'] /= decals['MW_TRANSMISSION_R']
        decals['FLUX_Z'] /= decals['MW_TRANSMISSION_Z']
        decals['FLUX_W1'] /= decals['MW_TRANSMISSION_W1']
        decals['FLUX_W2'] /= decals['MW_TRANSMISSION_W2']
        
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
    
    if mask_stars:
        x = decals['rmag']-decals['zmag']
        y = decals['rmag']-decals['w1mag']
        # Require valid grzW1W2 photometry and remove stars (log10(z)<-2)
        mask = np.isfinite(decals['gmag']) & np.isfinite(decals['rmag']) & np.isfinite(decals['zmag']) & \
                np.isfinite(decals['w1mag']) & np.isfinite(decals['w2mag']) & (y>2.5*x-2.5)
        decals = decals[mask]
        mgc = mgc[mask]

    # Calculate g-r, r-z, z-w1, w1-w2 color
    decals['gminr'] = decals['gmag']-decals['rmag']
    decals['rminz'] = decals['rmag']-decals['zmag']
    decals['zminw1'] = decals['zmag']-decals['w1mag']
    decals['w1minw2'] = decals['w1mag']-decals['w2mag']

    catalog = decals

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

    if mask_chisq0:
        catalog = catalog[mask_chisq]
    if mask_rad30:
        catalog = catalog[catalog.radius<30]

    return catalog
