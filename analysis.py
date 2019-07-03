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
import multiprocessing

def do_model(theta, args):
    DSnfw = DStheo(theta, args)
    return DSnfw

def find_best_fit(args, blind):
    z = args['z_mean']
    h = args['h']
    runtype = args['runtype']
    bestfitpath = args['bf_file']
    mean_mustar = args['mean_mustar']
    runconfig = args['runconfig']
    guess = get_model_start(runtype, mean_mustar, h, runconfig)
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
    nwalkers, nsteps = 64, 10000
    ndim = chain.shape[2]
    samples = chain.reshape((nwalkers, nsteps, ndim)) #Back to original chain shape
    #Saving blind chains
    blind_chainpath= chainpath.replace('.txt', '.npy').replace('chain_', 'blinded_chain_')
    print '-------- blind path:', blind_chainpath
    np.save(blind_chainpath, samples)

def do_mcmc(args, blind):
    nwalkers, nsteps = 64, 10000
    runtype   = args['runtype']
    bfpath    = args['bf_file']
    chainpath = args['chainfile']
    likespath = args['likesfile']
    runconfig = args['runconfig']
    binrun    = args['binrun']
    print '-- Chainpath:', chainpath
    if blind:
        bfmodel = pickle.load(open(bfpath.replace('.txt', '.p'), "rb"))
    else:
        bfmodel = np.loadtxt(bfpath) #Has everything
    start = get_mcmc_start(bfmodel, runtype, runconfig)
    ndim = len(start) #number of free parameters
    print 'ndim=', ndim
    if np.shape(start)==(1,1):
        start=np.reshape(start,(1,))
 
    pos = [ np.array(start) * 1.e-1 * abs(np.random.randn((ndim))) for k in range(nwalkers)] # work around for no dynamical range
    print 'Start points for mcmc : ', pos

    nproc = multiprocessing.cpu_count()
    threads = int(nproc/2.)
    print "Machine has nproc=", nproc, ". MCMC Will use ", threads, "threads!"

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, args=(args,), a=2, threads=threads)
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
        blindpath = '/data/des61.a/data/mariaeli/y1_wlfitting/ProfileFitting/blinding_file.p'
        make_blind_chain(chain, chainpath, blindpath)
    else:
        print 'Saving chains...'
        np.save(chainpath, sampler.chain)

    print 'Like path = ', likespath
    np.save(likespath, sampler.lnprobability)
    sampler.reset()
    return

if __name__ == "__main__":

    runtype = sys.argv[1]
    binrun  = int(sys.argv[2])
    runconfig = sys.argv[3]       #can be 'Full', 'OnlyM', 'FixAm'
    svdir = '_'+runconfig         #the suffix for the chains savedir
    if runtype=='cal' or runtype=='calsys':
        blind = True              #Temporaly changing to blind=True for sims, to test the unblinding scheme
    elif runtype=='data': 
        blind = True              #True for data
    cmodel='diemer18'             #'diemer18'
    factor2h = True               #True=2halo x h; False=2halo
    allrangeR = False             #True: r>0.2; False: 0.2<r<2.5 [in physical Mpc] 
    twohalo = False               #twohalo=True to compute the VERY SLOW 2-halo term with colossus

    dsfiles     = []
    dscovfiles  = []
    sampfiles   = []
    bfiles      = []
    bcovfiles   = []
    zmubins     = []

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

    elif runtype=='cal' or runtype=='calsys':

        zs = [0.0, 0.25, 0.5, 1.0] # redshifts of snapshots
        snap = [3,2,1,0]

        for zi in range(4):
            z = zs[zi]
            sn = snap[zi] # snapshot number
            for lj in range(4):
                if runtype=='cal':
                    dsfiles.append("deltasigma_z%d_m%d.txt"%(sn,lj))           #no systematics  #in comoving
                elif runtype=='calsys': 
                    dsfiles.append("deltasigma_z%d_m%d_phys_sys.txt"%(sn,lj))  #with systematics #in physical
                if sn==0:
                    zstr='zhigh2'
                    dscovfiles.append('y1mustar_y1mustar_qbin-%d-%d_dst_cov.dat'%(2, lj)) #data_z = [0,1,2]=[zlow, zmid, zhigh]; sim_z = [3, 2, 1, 0] = [zhigh2, zhigh, zmid, zlow]
                    bfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost.dat'%(2, lj))
                    bcovfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost_cov.dat'%(2, lj))
                    sampfiles.append('lgt20_mof_cluster_smass_full_coordmatch_p_central_star_%s_m%d.fits'%('zhigh',lj+1)) #assign bin zhigh data to zhigh2 on sims
                elif sn==1:
                    zstr='zhigh'
                    dscovfiles.append('y1mustar_y1mustar_qbin-%d-%d_dst_cov.dat'%(2, lj)) #reapeting the covariances from zhigh data to zhigh2 on sims
                    bfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost.dat'%(2, lj))
                    bcovfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost_cov.dat'%(2, lj))      
                    sampfiles.append('lgt20_mof_cluster_smass_full_coordmatch_p_central_star_%s_m%d.fits'%(zstr,lj+1))  
     	       	elif sn==2:
       	       	    zstr='zmid'
       	       	    dscovfiles.append('y1mustar_y1mustar_qbin-%d-%d_dst_cov.dat'%(1, lj))
                    bfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost.dat'%(1, lj))
                    bcovfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost_cov.dat'%(1, lj))
                    sampfiles.append('lgt20_mof_cluster_smass_full_coordmatch_p_central_star_%s_m%d.fits'%(zstr,lj+1))
     	       	elif sn==3:
       	       	    zstr='zlow'
       	       	    dscovfiles.append('y1mustar_y1mustar_qbin-%d-%d_dst_cov.dat'%(0, lj))
                    bfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost.dat'%(0, lj))
                    bcovfiles.append('y1mustar_y1mustar_qbin-%d-%d_zpdf_boost_cov.dat'%(0, lj))
                    sampfiles.append('lgt20_mof_cluster_smass_full_coordmatch_p_central_star_%s_m%d.fits'%(zstr,lj+1))
                
                sn_inv = snap[::-1][zi]
                zmubins.append([sn_inv,lj])   

    print dsfiles[binrun], '\n', dscovfiles[binrun], '\n', bfiles[binrun], '\n', bcovfiles[binrun]
    print '=== zmubins :', zmubins[binrun]

    #Get args and quantities
    start = time.time()
    if runtype=='data':
        args = get_args(dsfiles[binrun], dscovfiles[binrun], sampfiles[binrun], bfiles[binrun], bcovfiles[binrun], binrun, zmubins[binrun], runtype, svdir, cmodel, factor2h, allrangeR,twohalo,runconfig)
    if runtype=='cal' or runtype=='calsys':
        args = get_args(dsfiles[binrun], dscovfiles[binrun], sampfiles[binrun], bfiles[binrun], bcovfiles[binrun], binrun, zmubins[binrun], runtype, svdir, cmodel, factor2h, allrangeR, twohalo,runconfig)
    end = time.time()
    print 'Time to run get_args:', end - start, 'seconds'

    #Check the model run
    #theta=(1.e14)
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
