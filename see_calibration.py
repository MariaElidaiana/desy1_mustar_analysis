"""
See the calibration curves.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
plt.rc("text", usetex=True, fontsize=24)
plt.rc("errorbar", capsize=3)

def calc_cal(mu, z):
    #C0, a , b   = [ 0.95455359, 0.03090107, -0.18034199]  #diemer18
    #dC0, da, db = [0.02349054, 0.04583875, 0.06991657]    #diemer18
    C0, a , b   = [0.97767676,  0.0424727, -0.23112437]   #diemer18 + sys
    dC0, da, db = [0.02925375, 0.05641144, 0.09145634]    #diemer18 + sys

    Cbf = C0 * (mu/5.16e12)**a * ((1+z)/1.5)**b
    
    if use_SigmaInt:
        Cbfe = [errs for _ in range(len(Cbf))]
    else:
        Cbfe = np.sqrt( (Cbf**2/C0**2) * dC0**2 + Cbf**2 * (np.log(mu/5.16e12)**2 * da**2 + np.log((1+z)/1.5)**2 * db**2 ) )
    return Cbf, Cbfe

def get_cal_curve(z):
    """
    Return four arrays:
    - mu_star values
    - C best fit values
    - +/- error bars
    """
    mu = np.linspace(1e12, 10e12, 100)
    Cbf, Cbfe = calc_cal(mu, z)
    return [mu, Cbf, Cbfe]

#Load the data
use_SigmaInt = True
color = ['firebrick', 'tomato', 'royalblue', 'darkslateblue']

mariapath = "/Users/maria/current-work/maria_wlcode/Fox_Sims/"
mass = np.loadtxt(mariapath + "mustar_ps25_masses_crit_diemer18.txt")

if use_SigmaInt:
    bfs  = np.loadtxt(mariapath + "mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/diemer18_sims_masses_crit.txt")
    errs = 0.0166
else:
    bfs  = np.loadtxt(mariapath + "mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/diemer18_sims_masses_crit.txt")
    errs = np.loadtxt(mariapath + "mcmc_fox_calibration/mcmc_fox_calibration_1e11M1e18_diemer18_corr/diemer18_sims_err_masses_crit.txt")

C = mass/bfs
if not use_SigmaInt:
    Cerr = C * (errs/bfs) # the same as Cerr = mass*(errs/bfs**2)

mus = np.loadtxt(mariapath + "better_simulated_profiles/mustar_ps25_mustars.txt")
zs = [1.0, 0.5, 0.25, 0.0]

#Do the plot
fig, axarr = plt.subplots(4, sharex=True)
for i in range(len(zs)):
    z = zs[i]
    hi = mus[i]>0.
    if use_SigmaInt:
        axarr[i].errorbar(mus[i,hi]/1.e12, C[i,hi], errs, marker='.', ls='', c=color[i])
        print 'C-data = ', C[i,hi]
    else:
        axarr[i].errorbar(mus[i,hi]/1.e12, C[i,hi], Cerr[i,hi], marker='o', ls='', c=color[i])

    print calc_cal(mus[i,hi], z)[0]
    print calc_cal(mus[i,hi], z)[1]

    mux, Cbf, Cbfe = get_cal_curve(z)
    axarr[i].fill_between(mux/1.e12, Cbf-Cbfe, Cbf+Cbfe, color=color[i],alpha=0.2, zorder=-1)
    axarr[i].axhline(y=1.0, c='k', ls='--', lw=1)
    axarr[i].plot(mux/1.e12, Cbf, color=color[i], ls='-')

    ylim = .32
    axarr[i].set_xlim(1, 10)
    axarr[i].set_ylim(0.75, 1.25)
    axarr[i].text(8, 0.8, r"$z=%.2f$"%zs[i], fontsize=16)
    axarr[i].set_yticks([0.8, 0.9, 1, 1.1, 1.2])
    axarr[i].tick_params(labelsize=16)

plt.subplots_adjust(hspace=0.1, bottom=0.2, left=0.2)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.xlabel(r"$\mu_{\star}\,[10^{12} M_{\odot}]$")
plt.ylabel(r"${\cal C}=\frac{M_{\rm true}}{M_{\rm obs}}$")
plt.show()
