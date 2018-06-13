#!/bin/env python
"""
.. module:: plot_output
:synopsis: Script to run the mcmc analysis on data and Fox simulations.
.. moduleauthor:: Maria Elidaiana <mariaeli@brandeis.edu>
"""

import sys
import numpy as np
import time
from helper import *
from models import *
from likelihoods import *
import scipy.optimize as op
import emcee

def do_model(theta, args):
    DSnfw = DStheo(theta, args)
    return DSnfw

def find_best_fit(args):
    z = args['z_mean']
    h = args['h']
    runtype = args['runtype']
    bestfitpath = args['bf_file']
    mean_mustar = args['mean_mustar']
    guess = get_model_start(runtype, mean_mustar, h)
    nll = lambda *args: -lnposterior(*args)
    print "Running best fit"
    result = op.minimize(nll, guess, args=(args,), tol=1e-2)
    print "Best fit being saved at :\n\t%s"%bestfitpath
    print "\tsuccess = %s"%result['success']
    print result
    np.savetxt(bestfitpath, result['x'])
    return

def do_mcmc(args):
    nwalkers, nsteps = 32,5000
    runtype = args['runtype']
    bfpath = args['bf_file']
    chainpath = args['chainfile']
    likespath = args['likesfile']
    print chainpath
    bfmodel = np.loadtxt(bfpath) #Has everything
    start = get_mcmc_start(bfmodel, runtype)
    ndim = len(start) #number of free parameters

    priors = ([1e12, 1e15])

    # Initialize m200, p_cen, B0 and Rs at the center of prior:
    mid = np.zeros(ndim)
    for i in range(ndim):
        mid[i] = (priors[2*i] + priors[2*i + 1])
    mid *= 0.5

    zeros = np.zeros((ndim, nwalkers))
    np.random.seed(0)
    pos=[]
    for k in range(ndim):
        zeros[k,:] = mid[k] + 1e-1 * np.random.normal(loc=0.0, scale=np.abs(mid[k]), size=nwalkers)

    for i in range(nwalkers):
        if runtype=='data':
            pos.append(np.array([zeros[0,i],zeros[1,i], zeros[2,i],zeros[3,i]]))
        elif runtype=='cal':
            pos.append(np.array([zeros[0,i]]))

    print 'ndim', ndim
    #pos = [start + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    print 'pos', pos


    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, args=(args,), threads=8)
    print "Starting MCMC, saving to %s"%chainpath
    sampler.run_mcmc(pos, nsteps)
    print "MCMC complete"
    print "Mean acceptance fraction: %.3f" % (np.mean(sampler.acceptance_fraction))
    np.save(chainpath, sampler.chain)
    np.save(likespath, sampler.lnprobability)
    #np.savetxt(chainpath, sampler.flatchain)
    #np.savetxt(likespath, sampler.flatlnprobability)
    return


if __name__ == "__main__":

    runtype = sys.argv[1]
    binrun  = int(sys.argv[2])

    dsfiles   = []
    dscovfiles  = []
    sampfiles = []

    if runtype=='data':
        for i in xrange(3):
            for j in xrange(4):
                dsfiles.append('y1mustar_y1mustar_qbin-%d-%d_profile.dat'%(i, j))
                dscovfiles.append('y1mustar_y1mustar_qbin-%d-%d_dst_cov.dat'%(i, j))
                if i==0:
                    zstr='zlow'
                    sampfiles.append('lgt20_mof_cluster_smass_full_coordmatch_p_central_star_%s_m%d.fits'%(zstr,j+1))
                elif i==1:
                    zstr='zmid'
                    sampfiles.append('lgt20_mof_cluster_smass_full_coordmatch_p_central_star_%s_m%d.fits'%(zstr,j+1))
                elif i==2:
                    zstr='zhigh'
                    sampfiles.append('lgt20_mof_cluster_smass_full_coordmatch_p_central_star_%s_m%d.fits'%(zstr,j+1))

    elif runtype=='cal':

        zs = [0.0, 0.25, 0.5, 1.0] # redshifts of snapshots
        snap = [3,2,1,0]

        for zi in range(4):
            z = zs[zi]
            sn = snap[zi] # snapshot number
            for lj in range(4):
                dsfiles.append("deltasigma_z%d_m%d.txt"%(sn,lj))
                dscovfiles.append(None)
                sampfiles.append(None)

    #Get args and quantities
    start = time.time()
    args = get_args(dsfiles[binrun], dscovfiles[binrun], sampfiles[binrun], binrun, runtype)
    end = time.time()
    print 'Time to run get_args:', end - start, 'seconds'

    '''
    #Check the model run
    theta=1e14
    start = time.time()
    ds = do_model(theta, args)
    end = time.time()
    print 'Time to run do_model:', end - start, 'seconds'
    '''
    #Do the minimizer step
    start = time.time()
    find_best_fit(args)
    end = time.time()
    print 'Time to run find_best_fit:', end - start, 'seconds'

    #Do the mcmc
    start = time.time()
    do_mcmc(args)
    end = time.time()
    print 'Time to run do_mcmc:', end - start, 'seconds'
