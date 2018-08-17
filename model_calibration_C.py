#!/bin/env python
"""
.. module:: model_calibration_C
:synopsis: Script to run the mcmc of `C=\frac{M_{true}}{M_{obs}}`
for the Fox simulations results. Adapted from:
https://github.com/tmcclintock/fox_calibration/blob/master/old_files/model_calibration.py
.. moduleauthor:: Maria Elidaiana <mariaeli@brandeis.edu>
"""

import numpy as np
import emcee, sys
import scipy.optimize as op
import corner
import matplotlib.pyplot as plt

def get_C_model(params, mu, z):
    C0, a, b, lnf = params
    return C0 * (mu/mupivot)**a * ((1.+z)/zpivot)**b

def lnprior(params):
    C0, a, b, lnf = params
    #if C0 < 0.0: return -np.inf
    if C0 > 0.0 and -10.0 < lnf < 1.0: return 0.0
    return -np.inf

def lnlike(params, C, Cerr, mu, z):
    C0, a, b, lnf = params
    Cmodel = get_C_model(params, mu, z)
    inv_sigma2 = 1.0/(Cerr**2 + Cmodel**2 * np.exp(2*lnf))
    return -0.5*(np.sum( (C-Cmodel)**2*inv_sigma2 - np.log(inv_sigma2) ) )

def lnprob(params, C, Cerr, mu, z):
    lnp = lnprior(params)
    if not np.isfinite(lnp): return -np.inf
    return lnp + lnlike(params, C, Cerr, mu, z)

if __name__ == "__main__":

    #Loading M_true, M_obs, errors of M_obs and mean mu_stars
    cut    = 0 #Using all mu_star binnings
    M_true = np.genfromtxt("/Users/maria/current-work/maria_wlcode/Fox_Sims/mustar_ps25_masses_crit.txt")[:,cut:]
    mu     = np.genfromtxt("/Users/maria/current-work/maria_wlcode/Fox_Sims/data_files/mustar_ps25_mustars.txt")[:,cut:]
    M      = np.genfromtxt("/Users/maria/current-work/maria_wlcode/Fox_Sims/duffy08_masses_crit.txt")[:,cut:]
    Me     = np.genfromtxt("/Users/maria/current-work/maria_wlcode/Fox_Sims/duffy08_err_masses_crit.txt")[:,cut:]

    #Computing C=M_true/M_obs and sigma_C=Ce
    C  = M_true/M
    Ce = C*(Me/M)
    zs = [1.0, 0.5, 0.25, 0.0]
    z  = np.copy(zs)
    for i in range(3):
        z = np.vstack((z, zs))
    z = np.array(z).T

    zpivot = 1.5       #Equals to (1 + z0), where z0=0.5
    mupivot = 5.16e+12 #Same pivot as the previous SDSS low_z mu_star calibration

    #Optimizer step
    lnprobargs = (C, Ce, mu, z)
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, [1.05, 0.2, 0.2, np.log(0.01)], args=lnprobargs)

    #Do the mcmc
    nwalkers = 8
    ndim = 4
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprobargs)
    nsteps = 10000
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.flatchain
    np.savetxt("../test_calchain_s_int.txt", chain) #Chains are saved without remove burn-in
    means = np.mean(chain[-9000:], 0)
    stds = np.std(chain[-9000:], 0)

    print 'Burn-in=', nsteps - len(chain[-9000:])
    print 'means C0 a b ln(f) = ', means
    print 'errors C0 a b ln(f) = ',stds
    print
    print 'mean f', np.exp(means[3])
    print 'error f', np.abs((np.exp(means[3])/means[3]) * stds[3])

    samples = sampler.chain[:, 1000:, :].reshape((-1, ndim))

    #Plot ln(f)
    c0_mcmc, a_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                          zip(*np.percentile(samples, [16, 50, 84],
                                                             axis=0)))
    rlims = [[0.85, 1.1], [-0.17, 0.17], [-0.5, 0.0], [-12., 0.0]]
    fig = corner.corner(samples, labels=['$C_0$',r'$\alpha$', r'$\beta$', '$\ln\,f$'],
                        smooth=1.75, show_titles=True, range=rlims)
    plt.savefig('../test_C_calibration_with_intrinsic_scatter_lnf.png')

    #Plot f
    samples[:, 3] = np.exp(samples[:, 3])
    c0_mcmc, a_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                          zip(*np.percentile(samples, [16, 50, 84],
                                                             axis=0)))

    rlims = [[0.85, 1.1], [-0.17, 0.17], [-0.5, 0.0], [-0.02, 0.03]]
    fig = corner.corner(samples, labels=['$C_0$',r'$\alpha$', r'$\beta$', '$f$'],
                        smooth=1.75, show_titles=True, range=rlims,  title_fmt='.3f')
    plt.savefig('../test_C_calibration_with_intrinsic_scatter_f.png')
