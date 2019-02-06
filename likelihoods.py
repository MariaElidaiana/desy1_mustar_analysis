#!/bin/env python
"""
.. module:: likelihoods
:synopsis: Likelihoods for the mcmc.
.. moduleauthor:: Maria Elidaiana <mariaeli@brandeis.edu>
"""

import sys
import numpy as np
from models import *

#likelihood
def lnlikelihood(theta, args):
    h=args['h']
    z_mean = args['z_mean']
    DSobs = args['DS']                           #physical [Msun/pc^2]
    iDScov = args['icov']                        #physical [Msun/pc^2]
    runtype = args['runtype']
    DSmod = DStheo(theta, args)                  #physical [Msun/pc^2]
    DSmod = np.reshape(DSmod, np.shape(DSobs))   #need to reshape

    DSdiff = DSobs - DSmod                       #physical [Msun/pc^2]
    iDScov = iDScov                              #physical [Msun/pc^2]

    lnlikeDS = -0.5*np.dot(DSdiff, np.dot(iDScov, DSdiff))

    if runtype=='cal':
        return lnlikeDS
    if runtype=='data':
        Bp1 = args['Bp1']
        iBcov = args['iBcov']
        boost_model = get_boost_model(theta, args)
        Xb = Bp1 - boost_model
        LLboost = -0.5*np.dot(Xb, np.dot(iBcov, Xb))
        return lnlikeDS + LLboost


#prior
def lnprior(theta, args):
    runtype = args['runtype']
    Am_prior = args['Am_prior']
    Am_prior_var = args['Am_prior_var']
    runconfig =	args['runconfig']

    if runtype=='data':
       if runconfig=='Full':
            m200, pcc, Am, B0, Rs  = theta
            #Tom+Maria priors
            if m200<1e11 or m200>1e18 or Am <= 0.0 or pcc<0.0 or pcc>1.0: return -np.inf
            if Rs<=0.0 or B0<0.0 or Rs>4.0 or B0>1.0: return -np.inf
            lnprior_A = (Am_prior - Am)**2/Am_prior_var
            return -0.5*(lnprior_A)
       elif runconfig=='OnlyM':
            m200 = theta[0]            
            if m200<1e11 or m200>1e18: 
                lnprior_m200 = -np.inf
            else:
                lnprior_m200 = 0.0
            return -0.5*lnprior_m200
       elif runconfig=='FixAm':
            m200, pcc, B0, Rs  = theta
            #Tom+Maria priors
            if m200<1e11 or m200>1e18:
                lnprior_m200 = -np.inf
            else:
                lnprior_m200 = 0.0 
            if pcc<0.0 or pcc>1.0: return -np.inf
            if Rs<=0.0 or B0<0.0 or Rs>4.0 or B0>1.0: return -np.inf
            return -0.5*lnprior_m200

    if runtype=='cal':
        m200 = theta
        if m200<1e11 or m200>1e18:  #m200 is M200c [Msun]
            lnprior_m200 = -np.inf
        else:
            lnprior_m200 = 0.0
        return -0.5*lnprior_m200

# gaussian prior for misc
def ln_gaussian_prior(theta, args):
    runtype=args['runtype']
    runconfig =	args['runconfig']
    if runtype=='data':
       	if runconfig=='Full':
            m200, pcc, Am, B0, Rs  = theta
       	elif runconfig=='OnlyM':
       	    m200 = theta
            pcc=0.75
       	elif runconfig=='FixAm':
       	    m200, pcc, B0, Rs  = theta

    p_mu = args["p_cen_star_mu"]
    p_sigma=args["p_cen_star_sigma"]
    return -0.5*((pcc - p_mu)*(pcc - p_mu)/(p_sigma**2))

# posterior
def lnposterior(theta, args):
    runtype = args['runtype']
    runconfig = args['runconfig']
    if runtype=='data':
        if runconfig=='Full': 
            m200, pcc, Am, B0, Rs  = theta
        elif runconfig=='OnlyM':
            m200 = theta[0]
        elif runconfig=='FixAm':
            m200, pcc, B0, Rs  = theta   
        pt = lnprior(theta, args) + ln_gaussian_prior(theta, args)
        if not np.isfinite(pt):
            return -np.inf
        return lnlikelihood(theta, args) + pt

    elif runtype=='cal':
        m200  = theta
        prior = lnprior(theta, args)
        if not np.isfinite(prior):
            return -np.inf
        likelihood = lnlikelihood(theta, args)
        return likelihood + prior
