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

calpath    = '/data/des61.a/data/mariaeli/y1_wlfitting/better_simulated_profiles/'          #in comoving, no systematics
calpathsys = '/data/des61.a/data/mariaeli/y1_wlfitting/better_simulated_profiles_with_sys/' #in physical, with systematics
y1datapath = '/data/des61.a/data/mariaeli/y1_wlfitting/DES_mu_star_wlmass/results_v2/'
samplepath = '/data/des61.a/data/mariaeli/y1_wlfitting/DES_mu_star_samples/'
bfpath     = '/data/des61.a/data/mariaeli/y1_wlfitting/'
zbiaspath  = '/data/des61.a/data/mariaeli/y1_wlfitting/maria_z_files/'

def get_args(dsfile, dscovfile, samplefile, bfile, bcovfile, binrun, zmubins, runtype, svdir, cmodel, factor2h, allranger, twohalo, runconfig):
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

        B0_prior, B0_prior_var, Rs_prior, Rs_prior_var = get_BR_prior(zmubins[0], zmubins[1])

        print 'Sigma_inv = ', Sigma_crit_inv
        print 'p_cen_star_mean', p_mu
        print 'p_cen_star_sigma', p_sigma
        print 'mustar_mean', mean_mbin

        zbin = samplefile.split('_')[9]
        mbin = samplefile.split('_')[10].replace('.fits','')
        print zbin, mbin

        R, ds, icov, cov, fitmask = get_data_and_icov(dsfile, dscovfile, runtype, allranger)
        Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(bfile, bcovfile, runtype)

        results_dir = bfpath + 'fitting_'+ dsfile.split('_')[0] + '_' + zbin + '_'+ mbin + svdir +'/'

        #Fix to run in thread mode/parallel
        try:
            os.makedirs(results_dir, 0755)
        except OSError as e:
            if e.errno == 17:  # errno.EEXIST
                os.chmod(results_dir, 0755)

        bf_file = results_dir + 'bf_' + dsfile.split('_')[0]+ '_' + zbin + '_'+ mbin +'.txt'
        chainfile = results_dir + 'chain_' + dsfile.split('_')[0] + '_' + zbin + '_'+ mbin +'.txt'
        likesfile = results_dir + 'like_' + dsfile.split('_')[0]  +'_' + zbin + '_'+ mbin #+'.txt'

    elif runtype=='cal' or 'calsys':
        h=cosmology['h']
        om=cosmology['om']
        cosmo=FlatLambdaCDM(H0=h*100, Om0=om)
        print "Running on simulations: but getting pmem_sigma, pmem_mu from data sample file"

        samppath = samplepath + samplefile
        sample = tbl.read(samppath)

        p_cen_star = np.array(sample['P_MEMBER'])
        p_mu = np.mean(p_cen_star)
        p_sigma = np.std(p_cen_star)

        mustarpath = calpath + 'mustar_ps25_mustars.txt'
        m0, m1, m2, m3 = np.genfromtxt(mustarpath, unpack=True)

        zbinstr = dsfile.split('_')[1]
        mbin = dsfile.split('_')[2].split('.')[0]

        if zbinstr=='z3':
            zbin='zlow'
            z_mean = 0.0 
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
            z_mean = 0.25 
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
            z_mean = 0.5 
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
            z_mean = 1.0 
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

        R, ds, icov, cov, fitmask = get_data_and_icov(dsfile, dscovfile, runtype, allranger)
        Rb, Bp1, iBcov, Bcov = get_boost_data_and_cov(bfile, bcovfile, runtype)
        
        try:
            Am_prior, Am_prior_var = get_Am_prior(zmubins[0], zmubins[1])
            Sigma_crit_inv = get_sigma_crit_inv(zmubins[0], zmubins[1])
            print 'zmubins for Am and Sigma_inv = ', zmubins[0], zmubins[1]
            B0_prior, B0_prior_var, Rs_prior, Rs_prior_var = get_BR_prior(zmubins[0], zmubins[1]) 
        except IndexError:
            if zmubins[0]==3:
                Am_prior, Am_prior_var = get_Am_prior(2, zmubins[1])
                Sigma_crit_inv = get_sigma_crit_inv(2, zmubins[1])
                print 'zmubins for Am and Sigma_inv for high2 = ', '2', zmubins[1]
                B0_prior, B0_prior_var, Rs_prior, Rs_prior_var = get_BR_prior(2, zmubins[1])

        print 'Sigma_inv = ', Sigma_crit_inv

        results_dir = bfpath + 'fitting_'+ dsfile.split('_')[0] + '_' + zbin + '_'+ mbin + svdir +'/'

        try:
            os.makedirs(results_dir, 0755)
        except OSError as e:
            if e.errno == 17:  # errno.EEXIST
                os.chmod(results_dir, 0755)

        bf_file = results_dir + 'bf_' +   zbin + '_'+ mbin +'.txt'
        chainfile = results_dir + 'chain_'  + zbin + '_'+ mbin #+'.txt'
        likesfile = results_dir + 'like_' + zbin + '_'+ mbin #+'.txt'

    cosmodict = get_cosmo_default(runtype)
     
    args = {"h":h, "cosmo":cosmo, "z_mean":z_mean, "runtype":runtype,
            "p_cen_star_mu":p_mu, "p_cen_star_sigma":p_sigma, "zbin":zbin,
            "mbin":mbin, "R":R, "DS":ds, "icov":icov, "cov":cov, "bf_file":bf_file,
            "chainfile":chainfile, "likesfile":likesfile, "mean_mustar":mean_mbin,
            "Rb":Rb, "Bp1":Bp1, "Bcov":Bcov, "iBcov":iBcov, "Am_prior":Am_prior,
            "Am_prior_var":Am_prior_var, "Sigma_crit_inv":Sigma_crit_inv,
            "cmodel":cmodel, "factor2h":factor2h, "cosmodict":cosmodict, 
            "twohalo":twohalo, "runconfig":runconfig, "binrun":binrun,
            "B0_prior":B0_prior, "B0_prior_var":B0_prior_var, "Rs_prior":Rs_prior,
            "Rs_prior_var":Rs_prior_var}

    return args

def get_data_and_icov(dsfile, dscovfile, runtype, allranger):
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
        Radial bins in units of physical :math:`[Mpc/h]` (data)
    ds: np.array
        :math:`\Delta\Sigma` in units of physical :math:`[hMpc/pc^2]` (data)
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
        R0, DSobs0, DSerr0, dsx, dsxe = np.genfromtxt(datapath, unpack=True) #R=physical[Mpc/h], DS=physical[hMsun/pc^2], but h=1
 
        print 'All data R=physical[Mpc/h], DS=physical[hMsun/pc^2] :\n', np.c_[R0,DSobs0]

        covpath = y1datapath + dscovfile
        cov0 = np.genfromtxt(covpath)

        mubin = dsfile.split('_')[2].split('-')[2]
        print 'mu_star bin =', mubin

        R0     = R0/h     #physical [Mpc]
        DSobs0 = DSobs0*h #physical [Msun/pc^2] 
        cov0   = cov0*h   #physical [Msun/pc^2]            

        #Apply radial limits to avoid 2-halo term
        if mubin=='0':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu1
        elif mubin=='1':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu2
        elif mubin=='2':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu3
        elif mubin=='3':
            fitmask = (R0>=0.2)&(R0<=2.5) #mu4

        R   = R0[fitmask]     #physical [Mpc]
        ds  = DSobs0[fitmask] #physical [Msun/pc^2]
        cov = cov0[fitmask]   #physical [Msun/pc^2]
        cov = cov[:,fitmask]

        print 'Data after radial cut for the mcmc R=physical[Mpc], DS=physical[Msun/pc^2]:\n', np.c_[R, ds]
        #Apply the Hartlap correction, to get an unbiased cov matrix estimator.
        #It boosts the covariance values.
        Njk = 100.
        D = len(R)
        cov = cov*((Njk-1.)/(Njk-D-2)) #physical [Msun/pc^2]
        icov = np.linalg.inv(cov)      #physical

    elif runtype=='cal' or runtype=='calsys':
        h = cosmology['h']
        print "Using h = ", h
        print "Calibration file: ", dsfile
        if runtype=='cal':
            calibpath = calpath + dsfile
        elif runtype=='calsys': 
            calibpath = calpathsys + dsfile
 
        R0, ds0 = np.genfromtxt(calibpath, unpack=True) #if 'cal' comoving [Mpc/h], [hMsun/pc^2], but if 'calsys' physical [Mpc], [h Msun/pc^2]

        if runtype=='cal': 
            print 'All data (in comoving [Mpc/h],[hMsun/pc2]):\n', np.c_[R0,ds0]
        if runtype=='calsys': 
            print 'All data (in physical [Mpc],[Msun/pc2]):\n', np.c_[R0,ds0]

        mubin = dsfile.split('_')[2].split('.')[0][1]
        print "mu_star bin = ", mubin

        zbinstr = dsfile.split('_')[1]
        print "z bin = ", zbinstr
        if zbinstr=='z3':
            zbin='zlow'
            covzbin = '0'
            z_mean = 0.0 
        if zbinstr=='z2':
            zbin='zmid'
            z_mean = 0.25 
            covzbin = '1'
        if zbinstr=='z1':
            zbin='zhigh'
            z_mean = 0.5 
            covzbin = '2'
        if zbinstr=='z0':
            zbin='zhigh2'
            z_mean = 1.0 
            covzbin = '2'

        print "z mean = ", z_mean

        if runtype=='cal':
            #The simulations are in comoving distances.
            #I need to convert radius and deltasigma from comoving to physical.
            R0 /= h*(1.+z_mean)      #converting comoving [Mpc/h] to physical [Mpc]
            ds0 *= h*(1.+z_mean)**2  #converting comoving [hMsun/pc^2] to physical [Msun/pc^2]
        if runtype=='calsys':
            #Do nothing, because is in physical coordinates already 
            R0 = R0
            ds0 = ds0

        if mubin=='0':
            if allranger:
                fitmask = (R0>=0.2)
            else: 
                fitmask = (R0>=0.2)&(R0<=2.5) #mu1
        elif mubin=='1':
            if allranger:
                fitmask = (R0>=0.2)
            else: 
                fitmask = (R0>=0.2)&(R0<=2.5) #mu2
        elif mubin=='2':
      	    if allranger:
       	       	fitmask = (R0>=0.2)          
       	    else:
                fitmask = (R0>=0.2)&(R0<=2.5) #mu3
        elif mubin=='3':
      	    if allranger:
       	       	fitmask = (R0>=0.2)          
       	    else:
                fitmask = (R0>=0.2)&(R0<=2.5) #mu4

        R = R0[fitmask]  #physical [Mpc]
        ds= ds0[fitmask] #physical [Msun/pc^2]

        print 'Sim-data after radial cut (in physical [Mpc],[Msun/pc2]) for the mcmc:\n', np.c_[R, ds]

        if runtype=='cal':
            #The simulations don't have errors. The covariances for the simulated
            #data have to be the data covariances. That's why I getting them here.
            dscovfile = 'y1mustar_y1mustar_qbin-%d-%d_dst_cov.dat'%(int(covzbin),int(mubin)) #in physical [hMsun/pc^2]^2
            covpath = y1datapath + dscovfile  #to get the covariances
        elif runtype=='calsys':
            dscovfile = 'y1mustar_y1mustar_qbin-%d-%d_dst_cov_masked.dat'%(int(covzbin),int(mubin)) #in physical [Msun/pc^2]^2
            covpath = calpathsys + dscovfile  #to get the covariances

        print 'Covariance file for sim: ', covpath #dscovfile
        cov = np.genfromtxt(covpath)
        print 'fitmask', fitmask
        cov = cov[fitmask]

        if runtype=='cal':
            #The data cov is in units of physical [hMsun/pc^2]^2
            #Have to multiply by h to be in units of physical [Msun/pc^2]^2
            cov = cov[:,fitmask]*h
        elif runtype=='calsys':
            cov = cov[:,fitmask] #is already in physical [Msun/pc^2]^2

        #Apply the Hartlap correction, to get an unbiased cov matrix estimator.
        #It boosts the covariance values.
        Njk = 100.
        D = len(R)
        cov = cov*((Njk-1.)/(Njk-D-2))  #physical [Msun/pc^2]^2
        icov = np.linalg.inv(cov)       #physical [Msun/pc^2]^2
        print 'Done.' 
    return R, ds, icov, cov, fitmask

def get_boost_data_and_cov(Bfile, Bcovfile, runtype):
    cosmology = get_cosmo_default(runtype)
    if runtype=='data':
        h=cosmology['h']
        print "Using h = ", h
        print "Y1 boost-factor data file: ", Bfile
    elif runtype=='cal' or runtype=='calsys':
        h=cosmology['h']
        print "Using h = ", h
        print "Y1 boost-factor data file for Sims: ", Bfile

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

    indices = (Rb > 0.2)*(Rb<10.0)
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
    print 'Boost Factor data after radial cut for the mcmc:\n', np.c_[Rb, Bp1]
    return Rb, Bp1, iBcov, Bcov

def get_mcmc_start(model, runtype, runconfig):
    params = []
    if runtype=="data" or runtype=="calsys":
        if runconfig=="Full":
            m200, pcc, Am, B0, Rs  = model
            params.append(m200)
            params.append(pcc)
            params.append(Am)
            params.append(B0)
            params.append(Rs)
        elif runconfig=="OnlyM":
            m200 = model
            params.append(m200)
        elif runconfig=="FixAm":
            m200, pcc, B0, Rs  = model
            params.append(m200)
            params.append(pcc)
            params.append(B0)
            params.append(Rs)        
    elif runtype=="cal":
        m200 = model
        params.append(m200)
    return params

def get_model_defaults(h):
    #Dictionary of default starting points for the best fit
    defaults = {'m200': 1e14, #m200 is in Msun/h, but h=1
                'pcc' : 0.75,
                'Am'  : 1.00, #Y1 approx.
                'B0'  : 0.07, #In sims: 0.07 for zhigh2 and zhigh #0.07 for the others
                'Rs'  : 2.5}  #In sims: 0.02 for zhigh2 and zhigh #2.5  for the others #physical [Mpc]
    return defaults

def get_model_start(runtype, mean_mustar, h, runconfig):
    defaults = get_model_defaults(h)
    m200_guess = 1.77e14*(mean_mustar/5.16e12)**(1.74) #SDSS low-z relation

    if runtype=="data" or runtype=="calsys":
        if runconfig=="Full":
            guess = [m200_guess,
                     defaults['pcc'],
                     defaults['Am'],
                     defaults['B0'],
                     defaults['Rs']]
      	elif runconfig=="OnlyM":
            guess = [m200_guess]
      	elif runconfig=="FixAm":
            guess = [m200_guess,
                     defaults['pcc'],
                     defaults['B0'],
                     defaults['Rs']]
    elif runtype == "cal":
        guess = [m200_guess]
    return guess

def get_cosmo_default(runtype):
    if runtype=='data':
        #The cosmology used in this analysis
        cosmo = {'h'      : 1.0,
                 'om'     : 0.3,
                 'ode'    : 0.7,
                 'ob'     : 0.049,
                 'ok'     : 0.0,
                 'sigma8' : 0.835,
                 'ns'     : 0.962,
                 'de_model' : 'lambda'}

    elif runtype=='cal' or runtype=='calsys':
        #Fox cosmology
        cosmo = {'h'      : 0.6704,
                 'om'     : 0.318,
                 'ode'    : 0.682,
                 'ob'     : 0.049,
                 'ok'     : 0.0,
                 'sigma8' : 0.835,
                 'ns'     : 0.962}
    return cosmo

def get_Am_prior(zi, lj):
    #Photoz calibration (1+delta)
    deltap1 = np.loadtxt(zbiaspath + "Y1_deltap1_maria.txt")[zi, lj]
    deltap1_var = np.loadtxt(zbiaspath + "Y1_deltap1_var_maria.txt")[zi, lj]
    #Shear calibration m
    m = 0.012
    m_var = 0.013**2
    Am_prior = deltap1 + m
    Am_prior_var = deltap1_var + m_var
    print "\nAm files at: ", zbiaspath + "Y1_deltap1_maria.txt"
    print "deltap1 = ", deltap1
    print "Am_var file at: ", zbiaspath + "Y1_deltap1_var_maria.txt"
    print "deltap1_var = ", deltap1_var
    print "Am prior: z, mu, Am, sigma_Am = ", zi, lj, Am_prior, np.sqrt(Am_prior_var)
    return Am_prior, Am_prior_var

def get_sigma_crit_inv(zi, lj):
    sigma_crit_inv = np.loadtxt(zbiaspath + "sigma_inv_maria.txt")[zi, lj] #in physical [pc^2 / M_\odot]
    return sigma_crit_inv

def get_BR_prior(zi, lj):
    boostfactor_bfdir = "mcmc_boostfactor_0Rs10_0B01_02Rb10_64walkers/"
    #Boost-factor priors
    B0_bf     = np.loadtxt(calpathsys + boostfactor_bfdir + "Y1_boostfactor_B0_bf.txt")[zi, lj]
    B0err_bf  = np.loadtxt(calpathsys + boostfactor_bfdir + "Y1_boostfactor_B0err_bf.txt")[zi, lj]
    Rs_bf     = np.loadtxt(calpathsys + boostfactor_bfdir + "Y1_boostfactor_Rs_bf.txt")[zi, lj]    #in physical [Mpc]
    Rserr_bf  = np.loadtxt(calpathsys + boostfactor_bfdir + "Y1_boostfactor_Rserr_bf.txt")[zi, lj] #in physical [Mpc] 

    print "\nB(R) files at: ", calpathsys + boostfactor_bfdir
    print "B(R) priors: z, mu, B0_bf, B0err_bf, Rs_bf, Rserr_bf = ", zi, lj, B0_bf, B0err_bf, Rs_bf, Rserr_bf
    B0var_bf = B0err_bf**2
    Rsvar_bf = Rserr_bf**2
    print "B(R) priors variance B0var_bf, Rsvar_bf = ", B0var_bf, Rsvar_bf   
    return B0_bf, B0var_bf, Rs_bf, Rsvar_bf
