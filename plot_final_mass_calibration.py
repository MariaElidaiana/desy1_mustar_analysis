"""
Plot final mass calibration.
"""
import numpy as np
import emcee, sys
import scipy.optimize as op
import matplotlib.pyplot as plt
import corner
from astropy.table import Table as tbl
import pandas
from astropy.cosmology import FlatLambdaCDM

plt.rcParams.update({'font.size': 20})
plt.rc("text", usetex=True)#, fontsize=20)
plt.rc("errorbar", capsize=3)

def get_M_model(mu, z):
    zpivot = 1.35
    mupivot = 5.16e+12
    
    params  = [1.33438367e+16, 4.39728801e-01, -6.38095712e-01] #Values for fit using mean(mu)
    eparams = [3.48108442e+14, 2.62825488e-02, 1.67627628e-01]  
    
    M0, Fm, Gz = params
    dM0, dFm, dGz = eparams
    Cbf  = M0 * (mu/mupivot)**Fm * ((1.+z)/zpivot)**Gz
    Cbfe = np.sqrt( (Cbf**2/M0**2) * dM0**2 + Cbf**2 * (np.log(mu/5.16e12)**2 * dFm**2 + np.log((1+z)/1.35)**2 * dGz**2 ) )
    return Cbf, Cbfe

def get_cal_curve(z):
    lam = np.linspace(1e12, 5e13, 100)
    Cbf, Cbfe = get_M_model(lam, z)
    return [lam, Cbf, Cbfe]

#Loading data
zbin   = ['zlow', 'zmid', 'zhigh']
mubin  = ['m1', 'm2', 'm3', 'm4']
color  = ['royalblue', 'gray', 'tomato']
labels = [r'$0.1 \le z<0.33$', r'$0.33 \le z<0.5$', r'$0.5\le z<0.65$']

mariapath  = '/Users/maria/current-work/maria_wlcode/Fox_Sims/'
samplepath = '/Users/maria/current-work/lambda_star/connor-data-final/runs_Feb19/DES_mu_star_samples_nomucut/'

z, mu=[],[]
for i in range(len(zbin)):
    for j in range(len(mubin)):
        samplefile = 'lgt20_mof_cluster_smass_full_coordmatch_p_central_star_'+zbin[i]+'_'+mubin[j]+'.fits'
        print "Y1 sample file: ", samplefile
        samppath = samplepath + samplefile
        sample = tbl.read(samppath)
        z_sample = np.array(sample['Z'])
        z_mean = np.mean(z_sample)
        print 'z_mean = ', z_mean
        z.append(z_mean)
        
        mustar = np.array(sample['MU'])
        mu_mean = np.mean(mustar)*1e10
        print 'mu_mean = ', mu_mean
        mu.append(mu_mean)
z  = np.array(z)
mu = np.array(mu)
z  = z.reshape((3, 4))
mu = mu.reshape((3, 4))

M = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/diemer18_calibrated_blinded_masses_crit_Full.txt")
Me = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/diemer18_calibrated_blinded_err_masses_crit_Full.txt") #from mcmc

#Covariance is not from jackniffe, so Hartlap_factor = 1. (or no correction applied) #tp is in log now
C_stat = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/C_stat.txt")
C_SplusPz = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/C_SplusPz.txt")
C_Full = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/C_Full.txt")
iC_stat = np.linalg.inv(C_stat)
iC_SplusPz = np.linalg.inv(C_SplusPz)
iC_Full = np.linalg.inv(C_Full)

#Computing the errors in mass from the covariance matrix
diagC_Full = np.diag(C_Full)
diagC_Full = diagC_Full.reshape(3,4)
MeC_Full   = np.sqrt(diagC_Full)

C    = M
Cerr = MeC_Full
lams = mu
zs   = z

fig, axarr = plt.subplots(1)
for i in range(len(zs)):
    z = zs[i]
    hi = lams[i]>0.
    axarr.errorbar(lams[i,hi], C[i,hi], Cerr[i,hi], marker='.', ls='', c=color[i], zorder=3, label=labels[i])
    lamx, Cbf, Cbfe = get_cal_curve( np.mean(z) )
    axarr.fill_between(lamx, Cbf-1*Cbfe, Cbf+1*Cbfe, color=color[i],alpha=0.2, zorder=-1)
    axarr.plot(lamx, Cbf, color=color[i], ls='-')
    ylim = .32
    axarr.set_xscale('log')
    axarr.set_yscale('log')
    axarr.set_xlim(1e12, 2e13)
    axarr.set_ylim(3e15, 3.5e16)
plt.legend(loc="upper left")
plt.xlabel(r"$\mu_{\star}\,[ \rm{M_{\odot}}]$")
plt.ylabel(r"$M_{\rm{200c}}\,[ \rm{ h^{-1} M_{\odot} }]$")
plt.show()
