#!/bin/env python
"""
.. module:: likelihoods
:synopsis: Likelihoods for the mcmc.
.. moduleauthor:: Maria Elidaiana <mariaeli@brandeis.edu>
"""

from __future__ import division
import sys
import numpy as np
from models import *

def lnlikelihood(theta, args):
    h       = args['h']
    z_mean  = args['z_mean']
    DSobs   = args['DS']       #physical [Msun/pc^2]
    iDScov  = args['icov']     #physical [Msun/pc^2]
    runtype = args['runtype']
    
    DSmod   = DStheo(theta, args) #physical [Msun/pc^2]
    DSmod   = np.reshape(DSmod, np.shape(DSobs) )

    DSdiff  = DSobs - DSmod       #physical [Msun/pc^2]
    iDScov  = iDScov              #physical [Msun/pc^2]

    lnlikeDS = -0.5*np.dot(DSdiff, np.dot(iDScov, DSdiff))

    if runtype=='cal':
        return lnlikeDS
    if runtype=='data' or runtype=='calsys':
        Bp1 = args['Bp1']
        iBcov = args['iBcov']
        boost_model = get_boost_model(theta, args)
        Xb = Bp1 - boost_model
        LLboost = -0.5*np.dot(Xb, np.dot(iBcov, Xb))
        return lnlikeDS + LLboost

def lnprior(theta, args):
    runtype      = args['runtype']
    Am_prior     = args['Am_prior']
    Am_prior_var = args['Am_prior_var']
    runconfig    = args['runconfig']
    binrun       = args['binrun']
    B0_prior     = args['B0_prior']
    B0_prior_var = args['B0_prior_var']
    Rs_prior     = args['Rs_prior']     #physical [Mpc]
    Rs_prior_var = args['Rs_prior_var'] #physical [Mpc]

    if runtype=='data' or runtype=='calsys':
       
       if runconfig=='Full':
            m200, pcc, Am, B0, Rs  = theta
            
            if m200<1.e11 or m200>1.e18 or Am <= 0.0 or pcc<0.0 or pcc>1.0: return -np.inf
            if runtype=='calsys' and binrun==8 or binrun==9 or binrun==10 or binrun==11 or binrun==12 or binrun==13 or binrun==14 or binrun==15: #zhigh2 and zhigh bins
                if Rs<=0.0 or B0<0.0 or Rs>1.0 or B0>1.0: return -np.inf
            else:  
                if Rs<=0.0 or B0<0.0 or Rs>10.0 or B0>1.0: return -np.inf
            
            lnprior_A   = (Am_prior - Am)**2/Am_prior_var
            lnprior_pcc = (0.75 - pcc)**2/0.08**2
            lnprior_B0  = (B0_prior - B0)**2/B0_prior_var
            lnprior_Rs  = (Rs_prior - Rs)**2/Rs_prior_var
       
            return -0.5*(lnprior_A + lnprior_pcc + lnprior_B0 + lnprior_Rs)

       elif runconfig=='OnlyM':
            m200 = theta[0]
            
            if m200<1e11 or m200>1e18: 
                lnprior_m200 = -np.inf
            else:
                lnprior_m200 = 0.0
            return -0.5*lnprior_m200

       elif runconfig=='FixAm':
            m200, pcc, B0, Rs  = theta
            
            if m200<1e11 or m200>1e18:
                lnprior_m200 = -np.inf
            else:
                lnprior_m200 = 0.0 

            if pcc<0.0 or pcc>1.0: return -np.inf
            lnprior_pcc = (0.75 - pcc)**2/0.08**2

            if Rs<=0.0 or B0<0.0 or Rs>4.0 or B0>1.0: return -np.inf
            lnprior_B0  = (B0_prior - B0)**2/B0_prior_var
            lnprior_Rs  = (Rs_prior - Rs)**2/Rs_prior_var 
            return -0.5*(lnprior_m200 + lnprior_pcc + lnprior_B0 + lnprior_Rs)

    if runtype=='cal':
        m200 = theta
        if m200<1e11 or m200>1e18:  #m200 is M200c [Msun] physical
            lnprior_m200 = -np.inf
        else:
            lnprior_m200 = 0.0
        return -0.5*lnprior_m200

def lnposterior(theta, args):
    runtype = args['runtype']
    runconfig = args['runconfig']
    
    if runtype=='data' or runtype=='calsys':
        if runconfig=='Full': 
            m200, pcc, Am, B0, Rs  = theta
        elif runconfig=='OnlyM':
            m200 = theta[0]
        elif runconfig=='FixAm':
            m200, pcc, B0, Rs  = theta

        pt = lnprior(theta, args)
        if not np.isfinite(pt):
            return -np.inf

        lnp = lnlikelihood(theta, args) + pt
 
        if not np.isfinite(lnp):
            return -np.inf
        return lnp

    elif runtype=='cal':
        m200  = theta
        prior = lnprior(theta, args)

        if not np.isfinite(prior):
            return -np.inf
        likelihood = lnlikelihood(theta, args)

        return likelihood + prior
