"""
Model the calibration.
"""
import numpy as np
import emcee
import scipy.optimize as op
import corner
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
plt.rc('text', usetex=True)
ORG = plt.get_cmap('OrRd')

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

#Loading the data
simsdir = "/Users/maria/current-work/maria_wlcode/Fox_Sims/"

Msim    = np.genfromtxt(simsdir + "mustar_ps25_masses_crit_diemer18.txt")              #M200c [Mpc] converted from sims
mu      = np.genfromtxt(simsdir + "better_simulated_profiles/mustar_ps25_mustars.txt") #mu_star [Msun] from sims
M       = np.genfromtxt(simsdir + "mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/diemer18_sims_masses_crit.txt")     #M200c [Msun]
Me      = np.genfromtxt(simsdir + "mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/diemer18_sims_err_masses_crit.txt") #M200c [Msun]

C = Msim/M    #C=M_true/M_obs
Ce = C*(Me/M)

zs = np.array([1.0, 0.5, 0.25, 0.0])
z = np.copy(zs)
for i in range(3):
    z = np.vstack((z, zs))
z = np.array(z).T

zpivot = 1.5       #zpivot = (1+z0) = (1+0.5)
mupivot = 5.16e+12 #Same pivot as the previous SDSS mu_star calibration

#Mcmc configuration
ndim = 4
nwalkers = 32
nsteps = 10000

#Minimizer step
lnprobargs = (C, Ce, mu, z)
nll        = lambda *args: -lnprob(*args)
result     = op.minimize(nll, [1.04, 0.03, 0.025, np.log(0.01)], args=lnprobargs)
print result

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=lnprobargs)
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler.run_mcmc(pos, nsteps)
chain = sampler.flatchain

np.savetxt("./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/calchain_s_int_range0-2_2-5.txt", chain) #Chains are saved without remove burn-in
chainpath = "./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/calchain_s_int_range0-2_2-5.npy"
np.save(chainpath, sampler.chain)

#Read the flatchain format
chain = np.load(chainpath)

#Apply burn-in
burn_in = 2000
chain   = chain[:, burn_in:, :]
samples = chain.reshape((-1, chain.shape[2]))

means = np.array([ np.mean(samples[:, 0]), np.mean(samples[:, 1]), np.mean(samples[:, 2]), np.mean(samples[:, 3]) ])
stds  = np.array([ np.std(samples[:, 0]), np.std(samples[:, 1]), np.std(samples[:, 2]), np.std(samples[:, 3]) ])

print 'means C0 a b ln(V) = ', means
print 'errors C0 a b ln(V) = ', stds
print 'mean V =', np.exp(means[3])
print 'error V=', np.abs((np.exp(means[3])/means[3]) * stds[3])
print 'sqrt(mean V) = sigma_int = ', np.sqrt(np.exp(means[3]))

#Plotting corner
#Plot ln(V) = ln(sigma_int^2)
c0_mcmc, a_mcmc, b_mcmc, lnV_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
bestfit_values = [c0_mcmc[0], a_mcmc[0], b_mcmc[0], lnV_mcmc[0]]
fig = corner.corner(samples, labels=['$C_0$',r'$\alpha$', r'$\beta$', '$\ln\,\sigma_{\mathcal{C} }^2$'],
                    bins=40, smooth=2,
                    show_titles=True,
                    quantiles=[0.16, 0.5, 0.84],
                    color=ORG(0.7),
                    levels=[0.16, 0.50, 0.84],
                    label_kwargs=dict(fontsize=20),
                    title_kwargs=dict(fontsize=15),
                    plot_contours=True,
                    fill_contours=True,
                    hist_kwargs={"histtype": 'stepfilled', "alpha": 0.5,
                    "edgecolor": "none"},
                    use_math_text=True, truths=bestfit_values, truth_color='skyblue',
                    **{'title_fmt':'.3f', 'plot_datapoints': False})

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)
plt.savefig('./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/C_calibration_with_intrinsic_scatter_lnV_range0-2_2-5_colored.png', bbox_inches="tight")

#Plot V
samples[:, 3] = np.sqrt(np.exp(samples[:, 3])) #sqrt(exp(\ln sigma_int^{2}))
c0_mcmc, a_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                      zip(*np.percentile(samples, [16, 50, 84],
                                                         axis=0)))

bestfit_values = [c0_mcmc[0], a_mcmc[0], b_mcmc[0], f_mcmc[0]]
rlims = [[0.9, 1.05], [-0.085, 0.17], [-0.45, 0.0], [-0.01,0.06]]
fig = corner.corner(samples, labels=['$C_0$',r'$\alpha$', r'$\beta$', '$\sigma_{\mathcal{C}}$'],
                    bins=40, smooth=2,
                    show_titles=True,
                    range=rlims,
                    quantiles=[0.16, 0.5, 0.84],
                    color=ORG(0.7),
                    levels=[0.16, 0.50, 0.84],
                    label_kwargs=dict(fontsize=26),
                    title_kwargs=dict(fontsize=20),
                    plot_contours=True,
                    fill_contours=True,
                    hist_kwargs={"histtype": 'stepfilled', "alpha": 0.5,
                    "edgecolor": "none"},
                    use_math_text=True, truths=bestfit_values, truth_color='skyblue',
                    **{'title_fmt':'.3f', 'plot_datapoints': False})

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)
plt.savefig('./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/C_calibration_with_intrinsic_scatter_f_range0-2_2-5_colored.png', bbox_inches="tight")

#without titles
fig = corner.corner(samples, labels=['$C_0$',r'$\alpha$', r'$\beta$', '$\sigma_{\mathcal{C}}$'],
                    bins=40, smooth=2,
                    show_titles=False,
                    range=rlims,
                    quantiles=[0.16, 0.5, 0.84],
                    color=ORG(0.7),
                    levels=[0.16, 0.50, 0.84],
                    label_kwargs=dict(fontsize=20),
                    title_kwargs=dict(fontsize=17),
                    plot_contours=True,
                    fill_contours=True,
                    hist_kwargs={"histtype": 'stepfilled', "alpha": 0.5,
                    "edgecolor": "none"},
                    use_math_text=True, truths=bestfit_values, truth_color='skyblue',
                    **{'title_fmt':'.3f', 'plot_datapoints': False})
for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)
plt.savefig('./mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/C_calibration_with_intrinsic_scatter_f_range0-2_2-5_notitle_colored.png', bbox_inches="tight")
