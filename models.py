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
import cluster_toolkit as ct
import clusterlensing
from clusterlensing import ClusterEnsemble
import numpy as np
from astropy.cosmology import FlatLambdaCDM

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
    runtype = args['runtype']
    if runtype == 'data':
        m200, pcc, Am, B0, Rs   = theta     # in physical M200c [Msun]
    elif runtype == 'cal':
        m200 = theta[0]

    h       = args['h']
    R       = args['R'] # in physical [Mpc]
    cosmo   = args['cosmo']
    z_mean  = args['z_mean']
    sigma_crit_inv = args['Sigma_crit_inv']

    #conc_spline = args['cspline']

    #On data, using Duffy2008 concentration-mass relation:
    func = mass_concentration.duffy_concentration
    nfw = NFW(m200, func(m200, z_mean, cosmo), z_mean, cosmology=cosmo, overdensity_type='critical')

    #Now, on data and sim, using Diemer2015 concentartion-mass relation:
    #Have to fix the data cosmology part.
    #c200=conc_spline(m200, z_mean)
    #nfw = NFW(m200, c200, z_mean, cosmology=cosmo, overdensity_type='mean')

    #For DeltaSigma calculation, data and sim, the radius has to be in physical [Mpc]
    if runtype=='cal':
        ds  = nfw.delta_sigma(R).value #DeltaSigma is in units of physical [M_sun/Mpc^2]
        return ds/1.e12 #physical [M_sun/pc^2]

    if runtype=='data':
        ds  = nfw.delta_sigma(R).value #DeltaSigma is in units of physical [M_sun/Mpc^2]
        sigma = nfw.sigma(R).value

        # Computing miscetering correction from data
        m200p = np.array([m200])
        z     = np.array([z_mean])


        cluster = ClusterEnsemble(z, cosmology=FlatLambdaCDM(H0=100, Om0=0.3), cm='Duffy')
        cluster.m200 = m200p #in physical M200c [Msun]

        rbins = R/h         # in physical [Mpc], since h=1
        misc_off = 0.4/h    # here in [Mpc], since h=1 #miscentering offsets in 0.4 Mpc/h according to Simet 2016 paper

        offsets = np.ones(cluster.z.shape[0])*misc_off
        cluster.calc_nfw(rbins, offsets=offsets)  # NFW with offset
        dsigma_offset = cluster.deltasigma_nfw.mean(axis =0) # physical [M_sun/pc**2], but h=1, so [h M_sun/pc**2]?
        DSmisc = dsigma_offset.value   # physical [hMsun/pc^2]?

        sigma_offset = cluster.sigma_nfw.mean(axis =0) # physical [M_sun/pc**2], but h=1, so [h M_sun/pc**2]?
        Smisc = sigma_offset.value   # physical [hMsun/pc^2]?

        ds = ds/1.e12  # in units of h Msun/pc^2 physical (because h=1, otherwise is M_sun/pc^2)
        sigma = sigma/1.e12

        full_Sigma = (1-pcc)*sigma + pcc*Smisc

        full_model = pcc*(np.array(ds))+(1-pcc)*DSmisc

        full_model *= Am #shear+photo-z bias correction

        #Note: R (rbins) and Rs are in physical [Mpc], need to be comoving [Mpc/h]
        boost_model = ct.boostfactors.boost_nfw_at_R(rbins*h*(1+z_mean), B0, Rs*h*(1+z_mean))
        full_model /= boost_model

        full_model /= (1-full_Sigma*sigma_crit_inv) #Reduced shear

    return full_model # in physical [M_sun/pc^2]


def get_boost_model(theta, args):
    m200, pcc, Am, B0, Rs = theta
    h = args['h']
    z_mean  = args['z_mean']
    Rb       = args['Rb'] # in physical [Mpc]
    Rb *= h*(1+z_mean) #Rb and Rs are in physical [Mpc], need to be in comoving [Mpc/h]
    Rs *= h*(1+z_mean)
    return ct.boostfactors.boost_nfw_at_R(Rb, B0, Rs)
