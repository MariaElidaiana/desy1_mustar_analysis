"""
Model the calibration.
"""
import numpy as np
import emcee
import scipy.optimize as op
import corner
import matplotlib.pyplot as plt

simsdir = "/Users/maria/current-work/maria_wlcode/Fox_Sims/"
Msim    = np.genfromtxt(simsdir + "mustar_ps25_masses_crit_diemer18.txt")              #M200c [Mpc] converted from sims
mu      = np.genfromtxt(simsdir + "better_simulated_profiles/mustar_ps25_mustars.txt") #mu_star [Msun] from sims
M       = np.genfromtxt(simsdir + "mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18/diemer18_sims_masses_crit.txt")    #M200c [Msun]
Me      = np.genfromtxt(simsdir + "mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18/diemer18_sims_err_masses_crit.txt") #M200c [Msun]

C = Msim/M    #C=M_true/M_obs
Ce = C*(Me/M) #or (M_true*Me)/M**2

zs = [1.0, 0.5, 0.25, 0.0]
z = np.copy(zs)
for i in range(3):
    z = np.vstack((z, zs))
z = np.array(z).T

print '-'*30, 'z'
print z
print '-'*30, 'C'
print C
print '-'*30, 'Cerr'
print Ce
print '-'*30

zpivot = 1.5       #zpivot = (1+z0) = (1+0.5)
mupivot = 5.16e+12 #Same pivot as the previous SDSS mu_star calibration

def get_C_model(params, mu, z):
    C0, a, b, lnf = params
    return C0 * (mu/mupivot)**a * ((1.+z)/zpivot)**b

def lnprior(params):
    C0, a, b, lnV = params #V = sigma_int^2; the variance of the intrinsic scatter
    #set flat prior
    if C0<0. or a < -10. or a > 10. or b < -10. or b > 10. or lnV < -10. or lnV > 10.:
        return -np.inf
    return 0.

def lnlike(params, C, Cerr, mu, z):
    #Likelihood with intrinsinc scatter based on: http://adrian.pw/blog/fitting-a-line/
    C0, a, b, lnV = params
    Cmodel = get_C_model(params, mu, z)
    V = np.exp(lnV) #V=sigma_int^2
    N = len(C)
    dC = C - Cmodel
    ivar = 1 / (Cerr**2 + V) # inverse-variance now includes intrinsic scatter
    return -0.5 * (N*np.log(2*np.pi) - np.sum(np.log(ivar)) + np.sum(dC**2 * ivar))

def lnprob(params, C, Cerr, mu, z):
    lnp = lnprior(params)
    if not np.isfinite(lnp): return -np.inf
    return lnp + lnlike(params, C, Cerr, mu, z)

#Minimizer step
lnprobargs = (C, Ce, mu, z)
nll = lambda *args: -lnprob(*args)
result = op.minimize(nll, [1.04, 0.03, 0.03, 0.], args=lnprobargs)
print result

#Mcmc configuration
ndim = 4
nwalkers = 32
nsteps = 10000
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprobargs)
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler.run_mcmc(pos, nsteps)
chain = sampler.flatchain
np.savetxt("./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18/calchain_s_int_range0-2_2-5.txt", chain) #Chains are saved without remove burn-in
chainpath = "./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18/calchain_s_int_range0-2_2-5.npy"
np.save(chainpath, sampler.chain)

#Read the flatchain format
chain = np.load(chainpath)

#Apply burn-in
burn_in = 3000
chain = chain[:, burn_in:, :]
samples = chain.reshape((-1, chain.shape[2]))

print '-'*30
print 'Burn-in=', burn_in
means = np.array([ np.mean(samples[:, 0]), np.mean(samples[:, 1]), np.mean(samples[:, 2]), np.mean(samples[:, 3]) ])
stds  = np.array([ np.std(samples[:, 0]), np.std(samples[:, 1]), np.std(samples[:, 2]), np.std(samples[:, 3]) ])

print 'means C0 a b ln(V) = ', means
print 'errors C0 a b ln(V) = ',stds
print
print 'mean V =', np.exp(means[3])
print 'error V=', np.abs((np.exp(means[3])/means[3]) * stds[3])
print
print 'sqrt(mean V) = sigma_int = ', np.sqrt(np.exp(means[3]))
print '-'*30

#Plotting corner
samples = sampler.chain[:, burn_in:, :].reshape((-1, ndim))

#Plot ln(V) = ln(sigma_int^2)
c0_mcmc, a_mcmc, b_mcmc, lnV_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
rlims = [[0.85, 1.05], [-0.15, 0.186], [-0.5, 0.15], [-13., -2.5]]
fig = corner.corner(samples, labels=['$C_0$',r'$\alpha$', r'$\beta$', '$\ln\,\sigma_{\mathcal{C} }^2$'],
                    smooth=1.75, show_titles=False, range=rlims, title_fmt='.3f',
                    label_kwargs=dict(fontsize=20), title_kwargs=dict(fontsize=15), use_math_text=True)
for ax in fig.get_axes():
    #ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.tick_params(axis='both', labelsize=14)
plt.savefig('./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18/C_calibration_with_intrinsic_scatter_lnV_range0-2_2-5.png')

#Plot V
samples[:, 3] = np.sqrt(np.exp(samples[:, 3])) #sqrt(exp(\ln sigma_int^{2}))
c0_mcmc, a_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                      zip(*np.percentile(samples, [16, 50, 84],
                                                         axis=0)))
rlims = [[0.85, 1.05], [-0.15, 0.186], [-0.5, 0.15], [-0.03, 0.1]]
fig = corner.corner(samples, labels=['$C_0$',r'$\alpha$', r'$\beta$', '$\sigma_{\mathcal{C}}$'],
                    smooth=1.75, show_titles=False, range=rlims,  title_fmt='.3f',
                    label_kwargs=dict(fontsize=20), title_kwargs=dict(fontsize=17), use_math_text=True)
for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)
plt.savefig('./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18/C_calibration_with_intrinsic_scatter_f_range0-2_2-5.png')
