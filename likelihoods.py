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
    DSobs = args['DS']             #physical [Msun/pc^2]
    iDScov = args['icov']          #physical [Msun/pc^2]

    #cov = args['cov']
    #DSerr = np.sqrt(cov.diagonal())
    #DSerr = DSobs/5.

    DSmod = DStheo(theta, args)  #physical [Msun/pc^2]
    DSdiff = DSobs - DSmod       #physical [Msun/pc^2]
    iDScov = iDScov              #physical [Msun/pc^2]

    #print 'DSobs, DSmod in physical [Msun/pc^2] =============> ', DSobs/h, DSmod/h,'\n'

    lnlikeDS = -0.5*np.dot(DSdiff, np.dot(iDScov, DSdiff))
    #lnlikeDS = -0.5*np.sum((DSdiff )**2/(DSerr**2))
    return lnlikeDS

#prior
def lnprior(theta, args):
    runtype = args['runtype']
    if runtype=='data':
        m200, pcc, b0, rs  = theta
    elif runtype=='cal':
        m200 = theta

    if (m200<1e12)|(m200>1e15): #m200 is M200c [Msun] physical
        lnprior_m200 = -np.inf
    else:
        lnprior_m200 = 0.0

    return lnprior_m200

# posterior
def lnposterior(theta, args):
    runtype = args['runtype']
    if runtype=='data':
        m200, pcc, b0, rs  = theta
    elif runtype=='cal':
        m200  = theta

    prior = lnprior(theta, args)
    if not np.isfinite(prior):
        return -np.inf
    likelihood = lnlikelihood(theta, args)
    return likelihood + prior
