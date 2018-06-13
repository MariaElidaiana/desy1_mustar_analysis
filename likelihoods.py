#!/bin/env python
"""
.. module:: plot_output
:synopsis: Likelihood for the mcmc.
.. moduleauthor:: Maria Elidaiana <mariaeli@brandeis.edu>
"""

import sys
import numpy as np
from models import *

#likelihood
def lnlikelihood(theta, args):
    h=args['h']
    z_mean = args['z_mean']
    DSobs = args['DS']              #physical
    iDScov = args['icov']           #physical
    DSmod = DStheo(theta, args)     #comoving
    DSmod *= h*(1+z_mean)**2        #physical
    DSdiff = DSobs - DSmod
    lnlikeDS = -0.5*np.dot(DSdiff, np.dot(iDScov, DSdiff))
    return lnlikeDS

#prior
def lnprior(theta, args):
    runtype = args['runtype']
    if runtype=='data':
        m200, pcc, b0, rs  = theta
    elif runtype=='cal':
        m200 = theta

    if (m200<1e12)|(m200>1e15):
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
