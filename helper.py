#!/bin/env python
"""
.. module:: helper
:synopsis: Script to get paths and load data for the mcmc analysis.
.. moduleauthor:: Maria Elidaiana <mariaeli@brandeis.edu>
"""

import sys
from astropy.table import Table as tbl
from matplotlib import pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import os
from colossus.halo import concentration
from colossus.cosmology import cosmology
from scipy.interpolate import interp2d

calpath    = '/Users/maria/current-work/maria_wlcode/Fox_Sims/data_files/'
y1datapath = '/Users/maria/current-work/lambda_star/connor-data-final/runs_Feb19/DES_mu_star_wlmass/results_v2/'
samplepath = '/Users/maria/current-work/lambda_star/connor-data-final/runs_Feb19/DES_mu_star_samples/'
bfpath     = '/Users/maria/current-work/maria_wlcode/Fox_Sims/'
zbiaspath  = '/Users/maria/current-work/maria_wlcode/Fox_Sims/maria_z_files/'

def get_args(dsfile, dscovfile, samplefile, bfile, bcovfile, binrun, zmubins, runtype):
    """
    Pack as a dictionary the :math:`\Delta\Sigma` data and other relevant quantities
    for the computation of the theoretical :math:`\Delta\Sigma_{NFW}`.

    Parameters
    ----------
    dsfile: str
        Name of the file containing the :math:`\Delta\Sigma` measurements from :code:`xpipe`
    dscovfile: str
        Name of the file containing the Jackniffe covariances from :code:`xpipe`
    samplefile: str
        Name of file containing the original stacking samples used as input by :code:`xpipe`

    Returns
    -------
    args: dict
        Dictionary containing the data and other parameter for the model computation
    """

    cosmology = get_cosmo_default(runtype)

    if runtype=='data':
        h=cosmology['h']
        om=cosmology['om']
        cosmo=FlatLambdaCDM(H0=h*100, Om0=om)

        #dsfile, dscovfile = dsfile, dscovfile
        print "Y1 sample file: ", samplefile
        samppath = samplepath + samplefile
        sample = tbl.read(samppath)
        z_sample = np.array(sample['Z'])
        z_mean = np.mean(z_sample)
        print 'z_mean = ', z_mean

        p_cen_star = np.array(sample['P_MEMBER'])
        p_mu = np.mean(p_cen_star)
        p_sigma = np.std(p_cen_star)

        mustar = np.array(sample['MU'])
        mean_mbin = np.mean(mustar)*1e10 #fixing the mu_star units

        Am_prior, Am_prior_var = get_Am_prior(zmubins[0], zmubins[1])
        Sigma_crit_inv = get_sigma_crit_inv(zmubins[0], zmubins[1])

        print 'Sigma_inv = ', Sigma_crit_inv

        print 'p_cen_star_mean', p_mu
        print 'p_cen_star_sigma', p_sigma
        print 'mustar_mean', mean_mbin

        zbin = samplefile.split('_')[9]
        mbin = samplefile.split('_')[10].replace('.fits','')
        print zbin, mbin

        R, ds, icov, cov, fitmask = get_data_and_icov(dsfile, dscovfile, runtype)
        Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(bfile, bcovfile, runtype)

        results_dir = bfpath + 'fitting_'+ dsfile.split('_')[0] + '_' + zbin + '_'+ mbin +'/'

        print '=====>', results_dir

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        bf_file = results_dir + 'bf_' + dsfile.split('_')[0]+ '_' + zbin + '_'+ mbin +'.txt'
        chainfile = results_dir + 'chain_' + dsfile.split('_')[0] + '_' + zbin + '_'+ mbin +'.txt'
        likesfile = results_dir + 'like_' + dsfile.split('_')[0]  +'_' + zbin + '_'+ mbin #+'.txt'

        print '==== bf_file', bf_file, '\n', chainfile, '\n', likesfile

    elif runtype=='cal':
        h=cosmology['h']
        om=cosmology['om']
        cosmo=FlatLambdaCDM(H0=h*100, Om0=om)
        print "Running on simulations: no sample file"

        mustarpath = calpath + 'mustar_ps25_mustars.txt'
        m0, m1, m2, m3 = np.genfromtxt(mustarpath, unpack=True)

        zbinstr = dsfile.split('_')[1]
        mbin = dsfile.split('_')[2].split('.')[0]

        if zbinstr=='z3':
            zbin='zlow'
            z_mean = 0.0 #0.215
            if mbin=='m0':
                mean_mbin=m0[3]
            if mbin=='m1':
                mean_mbin=m1[3]
            if mbin=='m2':
                mean_mbin=m2[3]
            if mbin=='m3':
                mean_mbin=m3[3]
        if zbinstr=='z2':
            zbin='zmid'
            z_mean = 0.25 #0.415
            if mbin=='m0':
                mean_mbin=m0[2]
            if mbin=='m1':
                mean_mbin=m1[2]
            if mbin=='m2':
                mean_mbin=m2[2]
            if mbin=='m3':
                mean_mbin=m3[2]
        if zbinstr=='z1':
            zbin='zhigh'
            z_mean = 0.5 #0.575
            if mbin=='m0':
                mean_mbin=m0[1]
            if mbin=='m1':
                mean_mbin=m1[1]
            if mbin=='m2':
                mean_mbin=m2[1]
            if mbin=='m3':
                mean_mbin=m3[1]
        if zbinstr=='z0':
            zbin='zhigh2'
            z_mean = 1.0 #0.575
            if mbin=='m0':
                mean_mbin=m0[0]
            if mbin=='m1':
                mean_mbin=m1[0]
            if mbin=='m2':
                mean_mbin=m2[0]
            if mbin=='m3':
                mean_mbin=m3[0]

        print 'z_mean = ', z_mean
        print zbin, mbin
        print 'mustar_mean = ', mean_mbin


        p_mu = None
        p_sigma = None
        R, ds, icov, cov, fitmask = get_data_and_icov(dsfile, dscovfile, runtype)
        Rb, Bp1, iBcov, Bcov = [],[],[],[]

        Am_prior, Am_prior_var = [],[]

        results_dir = bfpath + 'fitting_'+ dsfile.split('_')[0] + '_' + zbin + '_'+ mbin +'/'

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        bf_file = results_dir + 'bf_' +   zbin + '_'+ mbin +'.txt'
        chainfile = results_dir + 'chain_'  + zbin + '_'+ mbin #+'.txt'
        likesfile = results_dir + 'like_' + zbin + '_'+ mbin #+'.txt'

    #conc_spline = get_concentration_spline(runtype) #not working for data

    args = {"h":h, "cosmo":cosmo, "z_mean":z_mean, "runtype":runtype,
            "p_cen_star_mu":p_mu, "p_cen_star_sigma":p_sigma, "zbin":zbin,
            "mbin":mbin, "R":R, "DS":ds, "icov":icov, "cov":cov, "bf_file":bf_file,
            "chainfile":chainfile, "likesfile":likesfile, "mean_mustar":mean_mbin,
            "Rb":Rb, "Bp1":Bp1, "Bcov":Bcov, "iBcov":iBcov, "Am_prior":Am_prior,
            "Am_prior_var":Am_prior_var, "Sigma_crit_inv":Sigma_crit_inv}
            #"cspline":conc_spline, "Rb":Rb, "Bp1":Bp1, "Bcov":Bcov, "iBcov":iBcov}

    return args

def get_data_and_icov(dsfile, dscovfile, runtype):
    """
    Get the :math:`\Delta\Sigma` and the covariance matrix from the data and apply
    the radial cut to minimize the effects of the 2-halo term.

    Parameters
    ----------
    dsfile: str
        Name of the file containing the :math:`\Delta\Sigma` measurements from :code:`xpipe`
    dscovfile: str
        Name of the file containing the jackknife covariances from :code:`xpipe`
    runtype: str
        Flag for use the data or the simulations

    Returns
    -------
    R: np.array
        Radial bins in units of :math:`[Mpc/h]`
    ds: np.array
        :math:`\Delta\Sigma` in units of :math:`[hMpc/pc^2]`
    icov: np.ndarray
        Inverse of the covariance matrix
    cov: np.ndarray
        Covariance matrix from jackknife
    fitmask: np.array
        Array with the boolean indexes after the radial binning cut

    Notes
    -----
    The :code:`runtype` flag can be :code:`data` for real data or :code:`cal` for
    the simulations.
    """
    cosmology = get_cosmo_default(runtype)
    if runtype=='data':
        h=cosmology['h']
        print "Using h = ", h
        print "Y1 data file: ", dsfile
        datapath = y1datapath + dsfile
        R0, DSobs0, DSerr0, dsx, dsxe = np.genfromtxt(datapath, unpack=True)
        print 'All data:\n', np.c_[R0,DSobs0]
        covpath = y1datapath + dscovfile
        cov = np.genfromtxt(covpath)

        mubin = dsfile.split('_')[2].split('-')[2]
        print 'mu_star bin =', mubin

        #Getting the same radial limits of the previous work to avoid 2-halo term
        #In the previus work I used the richeness-dependent relation of Simet et
        #al. 2017 to select the upper radial limits in each bin of richness.
        #The relation is 2.5*(lambda/20)**(1/3), choosing the lower-bin lambda of
        #the binning of that analysis, i.e., lambda=[20, 30, 40, 55] for each
        #richness bin, respectvely. In that work, I was also using lambda for the
        #mass-calibration for a comparison with mu_star. Then, it seemed to be ok
        #to keep the same radial range cuts for mu_star, for a fair comparison.
        #In Simet et al. the units are in physical. So, for the simulation data,
        #maybe I should convert this radial scale limits to comoving (?).
        if mubin=='0':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu1
        elif mubin=='1':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu2
        elif mubin=='2':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu3
        elif mubin=='3':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu4

        R = R0[fitmask]/h      #physical [Mpc]
        ds = DSobs0[fitmask]*h #physical [Msun/pc^2]
        cov = cov[fitmask]*h
        cov = cov[:,fitmask]
        print 'Data after radial cut for the mcmc:\n', np.c_[R, ds]
        #Apply the Hartlap correction, to get an unbiased cov matrix estimator.
        #It boosts the covariance values.
        Njk = 100.
        D = len(R)
        cov = cov*((Njk-1.)/(Njk-D-2)) #physical [Msun/pc^2]
        icov = np.linalg.inv(cov)      #physical

    elif runtype=='cal':
        print "Calibration file: ", dsfile
        calibpath = calpath + dsfile

        R0, ds0 = np.genfromtxt(calibpath, unpack=True) #comoving [Mpc/h], [hMsun/pc^2]
        print 'All data (in comoving [Mpc/h],[hMsun/pc2]):\n', np.c_[R0,ds0]

        mubin = dsfile.split('_')[2].split('.')[0][1]
        print "mu_star bin = ", mubin

        zbinstr = dsfile.split('_')[1]
        print "z bin = ", zbinstr
        if zbinstr=='z3':
            zbin='zlow'
            covzbin = '0'
            z_mean = 0.0 #0.215
        if zbinstr=='z2':
            zbin='zmid'
            z_mean = 0.25 #0.415
            covzbin = '1'
        if zbinstr=='z1':
            zbin='zhigh'
            z_mean = 0.5 #0.575
            covzbin = '2'
        if zbinstr=='z0':
            zbin='zhigh2'
            z_mean = 1.0 #0.575
            covzbin = '2'

        print "z mean = ", z_mean

        if mubin=='0':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu1
        elif mubin=='1':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu2
        elif mubin=='2':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu3
        elif mubin=='3':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu4

        R = R0[fitmask]   ##comoving [Mpc/h]      # not physical [Mpc]!
        ds = ds0[fitmask] ##comoving [hMsun/pc^2] # not physical [Msun/pc^2]!

        #The simulations are in comoving distances.
        #I need to convert radius and deltasigma from comoving to physical.
        h = cosmology['h']
        R /= h*(1.+z_mean)       #comoving [Mpc/h] to physical [Mpc]
        ds *= h*(1.+z_mean)**2  #comoving [hMsun/pc^2] to physical [Msun/pc^2]

        print 'Sim-data after radial cut (in physical [Mpc],[Msun/pc2]) for the mcmc:\n', np.c_[R, ds]

        #The simulations don't have errors. The covariances for the simulated
        #data have to be the data covariances. That's why I getting them here.
        dscovfile = 'y1mustar_y1mustar_qbin-%d-%d_dst_cov.dat'%(int(covzbin),int(mubin))
        covpath = y1datapath + dscovfile  #to get the covariances
        print 'Covariance file for sim: ', dscovfile
        cov = np.genfromtxt(covpath)

        cov = cov[fitmask]
        #The data cov is in units of physical [hMsun/pc^2]^2
        #Have to multiply by h to be in units of physical [Msun/pc^2]^2
        cov = cov[:,fitmask]*h

        #Apply the Hartlap correction, to get an unbiased cov matrix estimator.
        #It boosts the covariance values.
        Njk = 100.
        D = len(R)
        cov = cov*((Njk-1.)/(Njk-D-2))  #physical [Msun/pc^2]^2
        icov = np.linalg.inv(cov)       #physical [Msun/pc^2]^2

        #print '----- sqrt (Covariance) -----', '\n', np.sqrt(cov)
        #dcov = np.sqrt(cov.diagonal())
        #print '------ sqrt(Diagonal) -----', '\n', dcov


    return R, ds, icov, cov, fitmask

def get_boost_data_and_cov(Bfile, Bcovfile, runtype):
    cosmology = get_cosmo_default(runtype)
    if runtype=='data':
        h=cosmology['h']
        print "Using h = ", h
        print "Y1 boost-factor data file: ", Bfile

        #Boost-factors files
        Bdatapath = y1datapath + Bfile
        Rb, Bp1, Be = np.genfromtxt(Bdatapath, unpack=True)

        # Boost-factors Covariances
        Bcovpath = y1datapath + Bcovfile
        Bcov = np.loadtxt(Bcovpath)

        Becut = Be > 1e-6

        Bp1 = Bp1[Becut]
        Rb  = Rb[Becut]/h # physical [Mpc]
        Be  = Be[Becut]
        Bcov = Bcov[Becut]
        Bcov = Bcov[:,Becut]

        indices = (Rb > 0.2)*(Rb < 999.)
        Bp1 = Bp1[indices]
        Rb  = Rb[indices]
        Be  = Be[indices]
        Bcov = Bcov[indices]
        Bcov = Bcov[:,indices]
        Njk = 100.
        D = len(Rb)
        Bcov = Bcov*((Njk-1.)/(Njk-D-2))
        iBcov = np.linalg.inv(Bcov)
        print "Boost data shapes: ",Rb.shape, Bp1.shape, Be.shape, Bcov.shape
    return Rb, Bp1, iBcov, Bcov


def get_mcmc_start(model, runtype):

    params = []
    if runtype=="data":
        m200, pcc, Am, B0, Rs  = model
        params.append(m200)
        params.append(pcc)
        params.append(Am)
        params.append(B0)
        params.append(Rs)
    elif runtype=="cal":
        m200 = model
        params.append(m200)
    return params

def get_model_defaults(h):
    #Dictionary of default starting points for the best fit
    defaults = {'m200': 1e14, #m200 is in Msun/h
                'pcc' : 0.75,
                'Am' : 1.02, #Y1 approx.
                'B0'  : 0.10,
                'Rs'  : 2.50} #Mpc/h physical
    return defaults

def get_model_start(runtype, mean_mustar, h):

    defaults = get_model_defaults(h)
    #m200_guess = defaults['m200'] #now is not in function of mu
    m200_guess = 1.77e14*(mean_mustar/5.16e12)**(1.74) #SDSS low-z relation

    if runtype == "data":
        guess = [m200_guess,
                 defaults['pcc'],
                 defaults['Am'],
                 defaults['B0'],
                 defaults['Rs']]
    elif runtype == "cal":
        guess = [m200_guess]
    return guess

def get_cosmo_default(runtype):

    if runtype=='data':
        #The cosmology used in this analysis
        cosmo = {'h'      : 1,
                 'om'     : 0.3,
                 'ode'    : 0.7,
                 'ob'     : 0.0, #Have to fix: this cosmology doens't work
                 'ok'     : 0.0, #for Diemer 2015!
                 'sigma8' : 0.835,
                 'ns'     : 0.962,
                 'de_model' : 'lambda'}

    elif runtype=='cal':
        #Fox cosmology
        cosmo = {'h'      : 0.6704,
                 'om'     : 0.318,
                 'ode'    : 0.682,
                 'ob'     : 0.049,
                 'ok'     : 0.0,
                 'sigma8' : 0.835,
                 'ns'     : 0.962}
    return cosmo

#Set up the Concentration spline
def get_concentration_spline(runtype):
    cosmo = get_cosmo_default(runtype)
    if runtype=='data':
        params = {'flat':True,'H0':cosmo['h']*100.,'Om0':cosmo['om'],
                  'Ob0':cosmo['ob'], 'de_model': cosmo['de_model'],
                  'sigma8':cosmo['sigma8'], 'ns':cosmo['ns']}
    elif runtype=='cal':
        params = {'flat':True,'H0':cosmo['h']*100.,'Om0':cosmo['om'],
                  'Ob0':cosmo['ob'],'sigma8':cosmo['sigma8'],'ns':cosmo['ns']}

    cosmology.addCosmology('myCosmo', params)
    cos = cosmology.setCosmology('myCosmo')

    N = 20
    M = np.logspace(12, 17, N)
    z = np.linspace(0.1, 0.65, N)
    c_array = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            c_array[j,i] = concentration.concentration(M[i],'200c',z=z[j],model='diemer15')
    return interp2d(M, z, c_array)


def get_Am_prior(zi, lj):
    #Photoz calibration (1+delta)
    deltap1 = np.loadtxt(zbiaspath + "Y1_deltap1_maria.txt")[zi, lj]
    deltap1_var = np.loadtxt(zbiaspath + "Y1_deltap1_var_maria.txt")[zi, lj]
    #Shear calibration m
    m = 0.012
    m_var = 0.013**2
    Am_prior = deltap1 + m
    Am_prior_var = deltap1_var + m_var
    print "Am prior: ", zi, lj, Am_prior, np.sqrt(Am_prior_var)
    return Am_prior, Am_prior_var

def get_sigma_crit_inv(zi, lj):
    sigma_crit_inv = np.loadtxt(zbiaspath + "sigma_inv_maria.txt")[zi, lj] #in physical [pc^2 / M_\odot]
    return sigma_crit_inv
