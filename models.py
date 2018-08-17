#!/bin/env python
"""
.. module:: models
:synopsis: Computes the :math:`\Delta\Sigma_{NFW}` profile in units of
           :math:`[M_{\odot}/pc^2]` physical.
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
    m200    = theta     # in physical M200c [Msun]
    h       = args['h']
    R       = args['R'] # in physical [Mpc]
    cosmo   = args['cosmo']
    z_mean  = args['z_mean']
    runtype = args['runtype']
    conc_spline = args['cspline']

    #On data, using Duffy2008 concentration-mass relation:
    func = mass_concentration.duffy_concentration
    nfw = NFW(m200, func(m200, z_mean, cosmo), z_mean, cosmology=cosmo, overdensity_type='critical')

    #Now, on data and sim, using Diemer2015 concentartion-mass relation:
    #Have to fix the data cosmology part.
    #c200=conc_spline(m200, z_mean)
    #nfw = NFW(m200, c200, z_mean, cosmology=cosmo, overdensity_type='mean')

    #For DeltaSigma calculation, data and sim, the radius has to be in physical [Mpc]
    if runtype=='data' or runtype=='cal':
        ds  = nfw.delta_sigma(R).value #DeltaSigma is in units of physical [M_sun/Mpc^2]
    return ds/1.e12 #physical [M_sun/pc^2]
