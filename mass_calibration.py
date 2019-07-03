"""
Mass calibration.
"""
import numpy as np
import emcee, sys
import scipy.optimize as op
import matplotlib.pyplot as plt
import corner
from astropy.table import Table as tbl
import pandas
from astropy.cosmology import FlatLambdaCDM
import matplotlib

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
plt.rc('text', usetex=True)
ORG = plt.get_cmap('OrRd')

def get_M_model(params, mu, z):
    M0, Fm, Gz  = params #M0 is in log10
    return M0 * (mu/mupivot)**Fm * ((1.+z)/zpivot)**Gz

def lnprior(params):
    M0, Fm, Gz = params
    if M0<1e11 or M0>1.e18 or Fm<-10 or Fm>10 or Gz<-10 or Gz>10: return -np.inf
    return 0.0

def lnlike(params, M, iC_M, mu, z):
    Mmodel = get_M_model(params, mu, z)
    Mdiff = M-Mmodel
    lnlikeM = -0.5 * np.dot(Mdiff.flatten(), np.dot(iC_M, Mdiff.flatten()))
    return lnlikeM

def lnprob(params, M, iC_M, mu, z):
    lnp = lnprior(params)
    if not np.isfinite(lnp): return -np.inf
    return lnp + lnlike(params, M, iC_M, mu, z)

def do_mcmc(M, iC, mu, z, Cname):

    nwalkers = 64
    ndim     = 3
    nsteps   = 10000
    burn_in  = 2000

    #Optimization for initial values
    lnprobargs = (M, iC, mu, z)
    nll        = lambda *args: -lnprob(*args)
    result     = op.minimize(nll, [1e14, 1.45, -0.3], args=lnprobargs)
    print result

    #Mcmc
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprobargs)
    pos     = np.array([result["x"] + 1e-2*np.random.randn(ndim) for i in range(nwalkers)])

    sampler.run_mcmc(pos, nsteps)
    chain = sampler.flatchain
    np.savetxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/blind_mass_flatprior_calchain_zpivot035_"+Cname+".txt", chain)
    chainpath = mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/blind_mass_flatprior_calchain_zpivot035_"+Cname+".npy"
    np.save(chainpath, sampler.chain)

    chain   = np.load(chainpath)
    chain   = chain[:, burn_in:, :]
    samples = chain.reshape((-1, chain.shape[2]))
    
    print '---', Cname
    means = np.array([ np.mean(samples[:, 0]), np.mean(samples[:, 1]), np.mean(samples[:, 2])   ])
    stds  = np.array([ np.std(samples[:, 0]), np.std(samples[:, 1]), np.std(samples[:, 2]) ])
    
    print 'means M0, Fm, Gz  = ', means
    print 'errors M0, Fm, Gz = ',stds
    print 'log10(M0) =', np.log10(means[0])
    print 'error log10(M0) =', 0.434*(stds[0]/means[0])
    
    c0_mcmc, a_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                          zip(*np.percentile(samples, [16, 50, 84],
                                                             axis=0)))
                                                             
    bestfit_values = [np.log10(c0_mcmc[0]), a_mcmc[0], b_mcmc[0]]
    samples[:, 0]  = np.log10(samples[:, 0]) #M0 to log10(M0)
    
    rlims = [[16.10, 16.15], [0.38, 0.50], [-1.05, -0.2]]
    plt.figure()
    fig = corner.corner(samples, labels=['$\log_{10}M_0$',r'$F_{\mu_\star}$', r'$G_z$'],
                        bins=40, smooth=2,
                        show_titles=True,range=rlims,
                        quantiles=[0.16, 0.5, 0.84],
                        color=ORG(0.7),
                        levels=[0.16, 0.50, 0.84],
                        label_kwargs={'fontsize': 26},
                        plot_contours=True,
                        fill_contours=True,
                        title_kwargs={"fontsize": 20},
                        hist_kwargs={"histtype": 'stepfilled', "alpha": 0.5,
                        "edgecolor": "none"},
                        use_math_text=True, truths=bestfit_values, truth_color='skyblue',
                        **{'title_fmt':'.3f', 'plot_datapoints': False})
    plt.savefig(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/blind_mass_calchain_flatprior_corner_zpivot035_"+Cname+".png", bbox_inches="tight")
    plt.clf()

#Load the data
zbin=['zlow', 'zmid', 'zhigh']
mubin=['m1', 'm2', 'm3', 'm4']

mupath     = '/Users/maria/current-work/'
mariapath  = '/Users/maria/current-work/maria_wlcode/Fox_Sims/'
samplepath = mupath + 'lambda_star/connor-data-final/runs_Feb19/DES_mu_star_samples_nomucut/'

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
        
        mustar  = np.array(sample['MU'])
        mu_mean = np.mean(mustar)*1e10 #fixing the mu_star units
        print 'mu_mean = ', mu_mean
        mu.append(mu_mean)

z=np.array(z)
mu=np.array(mu)
z=z.reshape((3, 4))
mu=mu.reshape((3, 4))

M  = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/diemer18_calibrated_blinded_masses_crit_Full.txt")
Me = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/diemer18_calibrated_blinded_err_masses_crit_Full.txt")

#Covariance is not from jackniffe, so Hartlap_factor = 1. (or no correction applied)
C_stat    = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/C_stat.txt")
C_SplusPz = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/C_SplusPz.txt")
C_Full    = np.genfromtxt(mariapath + "mass_fitting_final_prior_0Rs10_0B01_02Rb10_1e11M1e18_D18_All/C_Full.txt")

iC_stat    = np.linalg.inv(C_stat)
iC_SplusPz = np.linalg.inv(C_SplusPz)
iC_Full    = np.linalg.inv(C_Full)

zpivot = 1.35      #means: z0+1 #zpivot = 0.35
mupivot = 5.16e+12 #same pivot as the previous SDSS low z, M x mu_star

#Run the mcmc
do_mcmc(M, iC_stat, mu, z, 'C_stat')
do_mcmc(M, iC_SplusPz, mu, z, 'C_SplusPz')
do_mcmc(M, iC_Full, mu, z, 'C_Full')

