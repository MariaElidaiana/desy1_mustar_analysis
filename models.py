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

from colossus.cosmology import cosmology
from colossus.halo import concentration
from colossus.halo import mass_defs
from colossus.halo import mass_adv
from colossus.halo import profile_outer
from colossus.halo import profile_nfw
from colossus.lss import bias

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
    runconfig = args['runconfig']

    if runtype == 'data':
        if runconfig=='Full':
            m200, pcc, Am, B0, Rs = theta #M200c [Msun]
        elif runconfig=='OnlyM':
            m200 = theta[0]
        elif runconfig=='FixAm':
            m200, pcc, B0, Rs = theta 
    elif runtype == 'cal':
        m200 = theta[0]

    h       = args['h']
    R       = args['R']     #in physical [Mpc]
    cosmo   = args['cosmo'] #astropy cosmology object
    z_mean  = args['z_mean']
    sigma_crit_inv = args['Sigma_crit_inv']
    cmodel  = args['cmodel']    #diemer18
    factor2h = args['factor2h'] #boolean, if True multiply 2-halo term by h 
    cosmodict = args['cosmodict']
    twohalo = args['twohalo']

    #Setting up the cosmology for Colossus
    params = {'flat':True,'H0':cosmodict['h']*100.,'Om0':cosmodict['om'],'Ob0':cosmodict['ob'],'sigma8':cosmodict['sigma8'],'ns':cosmodict['ns']}
    cosmology.addCosmology('myCosmo', params)
    cosmoc = cosmology.setCosmology('myCosmo')
    cmodel = cmodel

    c200c = concentration.concentration(m200*h, '200c', z_mean, model=cmodel, conversion_profile='nfw') #m200c_in is [Msun/h], m200c_out is [Msun/h]
    nfw   = NFW(m200, c200c, z_mean, cosmology=cosmo, overdensity_type='critical')                      #input mass should be in [Msun]
                    
    #For DeltaSigma calculation, data and sim, the radius has to be in physical [Mpc]
    if runtype=='cal':
        #ds  = nfw.delta_sigma(R).value #DeltaSigma is in units of physical [M_sun/Mpc^2]
        ds  = (nfw.delta_sigma(R).value)/1.e12   #DeltaSigma is in units of physical [M_sun/Mpc^2]

        if twohalo:
            #Adding the 2-halo term
            b = bias.haloBias(m200*h, z_mean, '200c', model='tinker10') #mass_in is [Msun/h] 
            #outer_term_mean = profile_outer.OuterTermMeanDensity(z = z_mean)
            outer_term_xi = profile_outer.OuterTermCorrelationFunction(z = z_mean, bias = b)
            p_nfw = profile_nfw.NFWProfile(M = m200*h, c = c200c, z = z_mean, mdef = '200c', outer_terms = [outer_term_xi]) #mass_in is [Msun/h] #outer_term_mean

            #Radius in physical kpc/h
            two_nfw0 = p_nfw.deltaSigmaOuter((R*1e3)*h, interpolate = True, interpolate_surface_density = False, min_r_interpolate=1.e-6*h, max_r_integrate=2.e5*h, max_r_interpolate=2.e5*h)
            two_nfw1 = two_nfw0/1.e6   #in physical [h Msun/pc^2]

            if factor2h: 
                two_nfw = h*(two_nfw1*h) #something like physical [Msun/(h pc^2)]
            else:
                two_nfw = (two_nfw1*h)   #in physical [Msun/pc^2] #This should be the right one
 
            ds_model = ds+two_nfw        #NFW + 2-halo in physical [Msun/pc^2] if factor2h=False
        else:
            ds_model = ds
        return ds_model #physical [M_sun/pc^2]

    if runtype=='data':

        ds    = (nfw.delta_sigma(R).value)/1.e12   #units of h Msun/pc^2 physical (but h=1, so actually is M_sun/pc^2)
        sigma = (nfw.sigma(R).value)/1.e12

        # Computing miscetering correction from data
        m200p = m200
        z     = np.array([z_mean])
        cluster = ClusterEnsemble(z,cosmology=FlatLambdaCDM(H0=100, Om0=0.3), cm='Diemer18', cmval=c200c) #, cm='Duffy')

        if np.shape(m200p)==(1,1):
            m200p = np.reshape(m200p, (1,))
 
        try:
            cluster.m200 = m200p #M200c [Msun]
        except TypeError:
            cluster.m200 = np.array([m200p])
         
        rbins = R          # in physical [Mpc]
        misc_off = 0.4     # here in [Mpc], since h=1 #miscentering offsets in 0.4 Mpc/h according to Simet 2016 paper
         
        offsets = np.ones(cluster.z.shape[0])*misc_off
        cluster.calc_nfw(rbins, offsets=offsets)             #NFW with offset

        dsigma_offset = cluster.deltasigma_nfw.mean(axis =0) #physical [M_sun/pc^2], but h=1, so [h M_sun/pc**2]?
        DSmisc = dsigma_offset.value                         #physical [hMsun/pc^2]?

        sigma_offset = cluster.sigma_nfw.mean(axis =0) #physical [M_sun/pc**2], but h=1, so [h M_sun/pc**2]?
        Smisc = sigma_offset.value                     #physical [hMsun/pc^2]?

        if runconfig=='OnlyM':
            pcc=0.75
            B0=0.50 
            Rs=2.00
   
        #The final model
        full_Sigma = pcc*sigma + (1-pcc)*Smisc

        full_model = pcc*ds + (1-pcc)*DSmisc
        
        if runconfig=='Full':
            full_model *= Am #shear+photo-z bias correction
        elif runconfig=='OnlyM' or 'FixAm':
            full_model = full_model

        #Note: R (rbins) and Rs are in physical [Mpc], need to be comoving [Mpc/h]
        boost_model = ct.boostfactors.boost_nfw_at_R(rbins*h*(1+z_mean), B0, Rs*h*(1+z_mean))
        full_model /= boost_model

        full_model /= (1-full_Sigma*sigma_crit_inv) #Reduced shear
        #print 'fullmodel=', full_model
        return full_model # in physical [M_sun/pc^2]


def get_boost_model(theta, args):
    runconfig=args['runconfig']
    if runconfig=='Full': 
        m200, pcc, Am, B0, Rs = theta
    elif runconfig=='OnlyM':
        m200 = theta#[0]
        B0=0.50
        Rs=2.00
    if runconfig=='FixAm':
        m200, pcc, B0, Rs = theta 

    h = args['h']
    z_mean  = args['z_mean']
    Rb       = args['Rb'] #in physical [Mpc]
    Rb *= h*(1+z_mean)    #Rb and Rs are in physical [Mpc], need to be in comoving [Mpc/h]
    Rs *= h*(1+z_mean)
    return ct.boostfactors.boost_nfw_at_R(Rb, B0, Rs)
