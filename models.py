#!/bin/env python
"""
.. module:: plot_output
:synopsis: Computes the :math:`\Delta\Sigma_{NFW}` profile in units of
           :math:`[h*M_{\odot}/pc^2]` comoving.
.. moduleauthor:: Maria Elidaiana <mariaeli@brandeis.edu>
"""

from __future__ import division
from NFW.nfw import NFW
from NFW import mass_concentration

def DStheo(theta, args):
    """
    Computes de theoretical :math:`\Delta\Sigma_{NFW}` profile for a given mass
    :math:`m_{200}`, z and cosmology.

    Parameters
    ----------
    theta: tuple
        Parameters for the mcmc
    args: dict
        Contains the cosmology and other mean values computed from the data

    Returns
    -------
    ds: np.array
        The thoretical :math:`\Delta\Sigma_{NFW}` profile

    Notes
    -----
    The parameters for the NFW function are the mass :math:`m_{200}` and the
    concentration :math:`c_{200}`. However, in this anlysis, we use the
    Duffy et al. (2008) concetration-mass relation to get the profile only
    as a function of the mass. See: https://github.com/joergdietrich/NFW for
    more details on the NFW function.
    """
    m200    = theta
    h       = args['h']
    R       = args['R']
    cosmo   = args['cosmo']
    z_mean  = args['z_mean']
    runtype = args['runtype']
    conc_spline = args['cspline']

    #On data, using Duffy2008 concentration-mass relation:
    #func = mass_concentration.duffy_concentration
    #nfw = NFW(m200, func(m200, z_mean, cosmo), z_mean, cosmology=cosmo)

    #Now, on data and sim, using Diemer2015 concentartion-mass relation:
    #Have to fix the data cosmology part.
    c200=conc_spline(m200, z_mean)
    nfw = NFW(m200, c200, z_mean, cosmology=cosmo)

    #Now, on data and sim, R has to be converted from physical to comoving
    if runtype=='data' or runtype=='cal':
        ds  = nfw.delta_sigma(R*h*(1+z_mean)).value
    return ds #comoving
