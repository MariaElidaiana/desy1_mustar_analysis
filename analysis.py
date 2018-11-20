#!/bin/env python
"""
.. module:: analysis
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
import pickle

def do_model(theta, args):
    DSnfw = DStheo(theta, args)
    return DSnfw

def find_best_fit(args, blind):
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
    if blind:
        print "\tsuccess = %s"%result['success']
        print "Blinded analysis. I'm not printing the results."
        pickle.dump(result['x'], open(bestfitpath.replace('.txt','.p'), "wb"))
    else:
        print "\tsuccess = %s"%result['success']
        print result
        np.savetxt(bestfitpath, result['x'])
    return


def make_blind(M, M_min, M_max, outpath):

    if os.path.isfile(outpath):
        alpha, c = pickle.load(open(outpath, "rb"))
    else:
        # Random blinding factors
        c = np.random.uniform(-0.5, 0.5)
        alpha = np.random.uniform(0.5,2.0)
        #Saving the blinding factors in a binary file
        outarr = np.array([alpha, c])
        pickle.dump(outarr, open(outpath, "wb"))

    # transform to (-1,1)
    x = 2*(M-M_min)/(M_max-M_min)-1
    # transform to -inf,inf
    x = np.arctanh(x)
    # rescale
    x = alpha*x + c
    # transform back to (-1,1)
    x = np.tanh(x)
    #transform back to full parameter range
    M_blind = 0.5*(1+x)*(M_max-M_min)+M_min
    return M_blind

def make_blind_chain(chain, chainpath, blindpath):
    samples = chain.reshape((-1, chain.shape[2])) #flatten the chain
    #Apply blinding
    samples[:, 0] = make_blind(samples[:, 0], 1.e11, 1.e18, blindpath)
    nwalkers, nsteps = 32, 10000
    ndim = chain.shape[2]
    samples = chain.reshape((nwalkers, nsteps, ndim)) #Back to original chain shape
    #Saving blind chains
    blind_chainpath= chainpath.replace('.txt', '.npy').replace('chain_', 'blinded_chain_')
    print '-------- blind path:', blind_chainpath
    np.save(blind_chainpath, samples)

def do_mcmc(args, blind):
    nwalkers, nsteps = 32, 10000 
    runtype = args['runtype']
    bfpath = args['bf_file']
    chainpath = args['chainfile']
    likespath = args['likesfile']
    print '-- Chainpath:', chainpath
    if blind:
        bfmodel = pickle.load(open(bfpath.replace('.txt', '.p'), "rb"))
    else:
        bfmodel = np.loadtxt(bfpath) #Has everything
    start = get_mcmc_start(bfmodel, runtype)
    ndim = len(start) #number of free parameters
    print 'ndim', ndim
    #pos = [start + 1e-3*np.random.randn(ndim) for k in range(nwalkers)]
    pos = [ np.array(start) * 1.e-1 * abs(np.random.randn((ndim))) for k in range(nwalkers)] # work around
    '''
    h0 = 1e-1
    np.random.seed(0)
    pos=[]
    zeros = np.zeros((ndim, nwalkers))
    for k in range(len(start)):
        zeros[k,:] = start[k] + h0 * np.random.normal(loc=0.0, scale=np.abs(start[k]), size=nwalkers)

    for i in range(nwalkers):
        pos.append(np.array([zeros[0,i]]))
    '''

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, args=(args,), a=2, threads=28)
    if blind:
        print "Starting MCMC, saving to %s"%chainpath.replace('.txt', '.npy').replace('chain_', 'blinded_chain_')
    else:
        print "Starting MCMC, saving to %s"%chainpath
    sampler.run_mcmc(pos, nsteps)
    print "MCMC complete"
    print "Mean acceptance fraction: %.3f" % (np.mean(sampler.acceptance_fraction))

    if blind:
        print 'Saving the blinded chains...'
        chain = sampler.chain
        #blindpath = '/Users/maria/current-work/maria_wlcode/Fox_Sims/ProfileFitting/blinding_file.p'
        blindpath = '/data/des61.a/data/mariaeli/y1_wlfitting/ProfileFitting/blinding_file.p'
        make_blind_chain(chain, chainpath, blindpath)
    else:
        print 'Saving chains...'
        np.save(chainpath, sampler.chain)

    print 'Like path = ', likespath#.replace('.txt','')
    np.save(likespath, sampler.lnprobability)
    sampler.reset()
    return


if __name__ == "__main__":

    runtype = sys.argv[1]
    binrun  = int(sys.argv[2])
    svdir= '_final' #'_v2'
    blind = True

    dsfiles   = []
    dscovfiles  = []
    sampfiles = []
    bfiles = []
    bcovfiles = []

    zmubins = []

    if runtype=='data':
        for i in xrange(3):
            for j in xrange(4):
                dsfiles.append('y1mustar_y1mustar_qbin-%d-%d_profile.dat'%(i, j))
                dscovfiles.append('y1mustar_y1mustar_qbin-%d-%d_dst_cov.dat'%(i, j))

                bfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost.dat'%(i, j))
                bcovfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost_cov.dat'%(i, j))

                zmubins.append([i,j])

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

    print dsfiles[binrun], '\n', dscovfiles[binrun], '\n', bfiles[binrun], '\n', bcovfiles[binrun]

    #Get args and quantities
    start = time.time()
    args = get_args(dsfiles[binrun], dscovfiles[binrun], sampfiles[binrun], bfiles[binrun], bcovfiles[binrun], binrun, zmubins[binrun], runtype, svdir)
    end = time.time()
    print 'Time to run get_args:', end - start, 'seconds'


    #Check the model run
    #theta=1e14
    #theta = (1e14, 0.8, 1.02, 0.33, 0.45)
    #start = time.time()
    #ds = do_model(theta, args)
    #end = time.time()
    #print 'Time to run do_model:', end - start, 'seconds'


    #Do the minimizer step
    start = time.time()
    find_best_fit(args, blind)
    end = time.time()
    print 'Time to run find_best_fit:', end - start, 'seconds'


    #Do the mcmc
    start = time.time()
    do_mcmc(args, blind)
    end = time.time()
    print 'Time to run do_mcmc:', end - start, 'seconds'
