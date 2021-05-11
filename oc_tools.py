# This file contains some functions written by Gergely Hajdu to derive
# O-C values from light curves using the modified Hertzsprung method,
# as decribed in Hajdu et al. (2021)

# We start by setting the maximum number of threads for Numpy to 1
# If we don't do this, the performance might suffer

import os
os.environ["OMP_NUM_THREADS"] = "1"

# We import some libraries we will be using here
# Note that using Numba is optional, the import and the @jit decorators can
# be commented out to disable the,

import numpy as np
import emcee
import matplotlib.pyplot as plt
from   matplotlib.colors import to_rgba
from   chainconsumer   import ChainConsumer as CC
from   scipy.optimize  import least_squares
from   sklearn         import linear_model
from   multiprocessing import Pool
from   tqdm            import tqdm
from   numba           import jit

# We will use this color later when plotting the 

colorC1 = (to_rgba('C1')[0],to_rgba('C1')[1],to_rgba('C1')[2],0.3)


# This function solves the LTTE equations

@jit(nopython=True)
def calc_orbit(e, P_orb, T_peri, a_sini, omega, times):
    if e<0.3:
        loop=10
    else:
        loop=int(16.1 - 46 * e + 86* e**2)
        
    times_mod = np.copy(times)
    if times_mod[0]-T_peri < 0.:
        times_mod += np.ceil((T_peri-times_mod[0])/P_orb)*P_orb
        
    M = 2 * np.pi * ( (( times_mod - T_peri ) / P_orb) - (( times_mod - T_peri ) // P_orb) )
    E = M
    for i in range(loop):
        E = M + e * np.sin(E)
        
    nu_part = np.arccos( (np.cos(E) - e) / (1 - e * np.cos(E)))
    nu = ( M // np.pi) * 2 * np.pi - nu_part + 2 * nu_part * ( 1 - (M // np.pi))
    LT = a_sini * ( (1 - e**2)/(1 + e * np.cos(nu)) * np.sin(nu + omega)  )

    return LT


# Function to calculate the semi-amplitude of the radial velocity

def calc_K(e, P_orb, T_peri, a_sini, omega):
    K  = (2 * np.pi * a_sini / (P_orb *np.sqrt(1-e**2))) * 299792.458 
    
    return K

# Function to calculate the mass function

def calc_mass_function(a_sini, P_orb):
    return ((a_sini**3)/(P_orb**2))* (173.14738217**3 * 365.2422**2)

# Function to prepare the design matrix X for a Fourier series

def return_harmonic_series(order, period, epochs):
    X=np.zeros((order*2, epochs.size))
    X[0::2]=np.sin( np.arange(1, order+1).reshape((-1,1)) * 2 * np.pi * epochs / period )
    X[1::2]=np.cos( np.arange(1, order+1).reshape((-1,1)) * 2 * np.pi * epochs / period )
    return X.T

# Function to calculate the value of a Fourier series at arbitrary points in time
# Used for plotting as well as calculating residuals

def return_harmonic_LC(order,period,coefs,intercept,positions):
    result=np.zeros(positions.size)
    for i in range(1,order+1):
        result=result +\
               coefs[i*2-2] *np.sin( i*2*np.pi*positions/period ) +\
               coefs[i*2-1] *np.cos( i*2*np.pi*positions/period)
    return result+intercept


# This function calculates the difference in mean intensity between OGLE-III
# and IV light curves, and matches to former to the latter

def shift_int(jd3, mag3, jd4, mag4, order, period, plot=False):
    int3 = 1/10**((mag3-15) * 0.4)
    X3 = return_harmonic_series(order, period, jd3)
    clf3=linear_model.LinearRegression()
    clf3.fit(X3, int3)

    int4 = 1/10**((mag4-15) * 0.4)
    if mag4.size>20:
        X4 = return_harmonic_series(order, period, jd4)
    else:
        X4 = return_harmonic_series(3, period, jd4)
    clf4=linear_model.LinearRegression()
    clf4.fit(X4, int4)
    int_diff = clf3.intercept_ - clf4.intercept_
    mag3_new = -2.5*np.log10(int3-int_diff)+15
    
    if plot:
        phases=np.arange(0.0,0.999,0.001)
        min_lc=np.min((mag3.min(),mag4.min()))-0.02
        max_lc=np.max((mag3.max(),mag4.max()))+0.02
        plt.figure(figsize=(14,12))
    
        ax1=plt.subplot2grid((3,2), (0,0), colspan=2)
        ax2=plt.subplot2grid((3,2), (1,0))
        ax3=plt.subplot2grid((3,2), (1,1), sharey=ax2)
        ax4=plt.subplot2grid((3,2), (2,0), colspan=2, sharey=ax1)

        ax1.set_ylim(max_lc, min_lc)
        ax2.set_xlim(0,1)
        ax3.set_xlim(0,1)
        fit3 = return_harmonic_LC(order,1,clf3.coef_,clf3.intercept_,phases)
        fit4 = return_harmonic_LC(order,1,clf4.coef_,clf4.intercept_,phases)
        
        ax1.plot(jd3,mag3,'.')
        ax1.plot(jd4,mag4,'C2.')
        ax2.plot(np.modf(jd3/period)[0],int3,'.')
        ax2.plot(phases,fit3,'-', lw=4.0)
        ax3.plot(np.modf(jd4/period)[0],int4,'C2.')
        ax3.plot(phases,fit4,'C1-', lw=4.0)
        ax4.plot(jd3,mag3_new,'.')
        ax4.plot(jd4,mag4,'C2.')
        
        plt.show()
    
    return mag3_new

# This function is used to split the light curve into seasons, then split those seasons
# into shorter sections 
# Note that the detection of season borders in automatized

def split_lc_seasons(jd,
                     limits       = np.array((0, 8, 80, 120, 160, 240, np.inf)),
                     into         = np.array((0, 1, 2,  3,   4,   5)),
                     granularity  = 10.,
                     plot         = False,
                     mag          = None):
    
    trials      = np.arange(0., 365, granularity)
    first_phase = np.zeros_like(trials)
    masks       = []
    indices     = np.arange(jd.size)

    for i in range(trials.size):
        jd_phases   = np.modf((jd-(jd.min()-trials[i]))/365.2422)[0]
        first_phase[i] = jd_phases.min()
        
    zero_epoch = jd.min()-trials[first_phase.argmax()]+first_phase[first_phase.argmax()]*365.2422/2
    
    no_seasons = np.int(np.ceil((jd.max()-zero_epoch)/365.2422))
    for i in range(no_seasons):
        t_mask = (jd > zero_epoch + i * 365.2422) * (jd < zero_epoch + (i+1) * 365.2422)
        
        cont = True
        j    = 0
        while cont:
            if (t_mask.sum() >= limits[j]) and (t_mask.sum() < limits[j+1]):
                if into[j]==0:
                    cont = False
                else:
                    for k in range(into[j]):
                        mask = np.zeros_like(jd, dtype=np.bool)
                        mask[np.array_split(indices[t_mask], into[j])[k]] = True
                        masks.append(mask)
                    cont = False
            else:
                j = j +1 
                
    masks = np.asanyarray(masks)
    
    if plot:
        fig = plt.figure(figsize=(14,10))

        ax1=plt.subplot2grid((2,1), (0,0), colspan=2)
        max_lc = mag.max()+0.03
        min_lc = mag.min()-0.03
        ax1.set_ylim(max_lc, min_lc)
        ax1.plot(jd,mag,'k.')
        
        for i in range(masks.shape[0]):
            ax1.plot(jd[masks[i]], mag[masks[i]],'.')
                    
    return masks


# This function calculates the residuals for a given shift (trial O-C value),
# which are minimized to determine the O-C values themselves

def return_residuals(shift, jd, mag, coefs, intercept, period, order):
    jd_s = jd - shift
    
    return mag - return_harmonic_LC(order,period,coefs,intercept,jd_s)

# This function determines the O-C values by first fitting the light curve,
# either for the original or the one with corrected timings from the first O-C fit,
# if provided through the "jd_mod" optional variable. Errors are also calculated
# and returned if the "bootstrap_times" optional variable is set to a positive
# integer value

def calc_oc_points(jd, mag, period, order, splits, bootstrap_times = 0, jd_mod = None, figure=False):
    oc_jd = np.zeros(splits.shape[0])
    oc_oc = np.zeros_like(oc_jd)
    oc_sd = np.zeros_like(oc_jd)
    
    if jd_mod is None:
        X = return_harmonic_series(order,period,jd)
    else:
        X = return_harmonic_series(order,period,jd_mod)

    clf = linear_model.LinearRegression()
    clf.fit(X, mag)
    
    if figure:
        phases=np.arange(0.0,0.999,0.001)
        fit = return_harmonic_LC(order,1,clf.coef_,clf.intercept_,phases)
        
        fig = plt.figure(figsize=(8,5))
        plt.gca().invert_yaxis()
        
        if jd_mod is None:
            plt.plot(np.modf(jd/period)[0],mag,'.')
        else:
            plt.plot(np.modf(jd_mod/period)[0],mag,'.')
        plt.plot(phases,fit,'-',lw=4.0)
        plt.xlim(0,1)
        plt.show()
    
    for i in tqdm(range(splits.shape[0]), ncols=80):
        oc_jd[i] = np.mean(jd[splits[i]])
        
        if i==0:
            initial_guess = np.asarray([0.0])
        
        lsq = least_squares(return_residuals, x0 = initial_guess,
                            args=(jd[splits[i]], mag[splits[i]], clf.coef_, clf.intercept_, period, order))
        oc_oc[i] = lsq.x
        initial_guess = lsq.x
        
        if bootstrap_times != 0:
            boot_sample_ids = np.arange(jd[splits[i]].size)
            boot_store = np.zeros((bootstrap_times))
            np.random.seed(0)
            
            for j in range(bootstrap_times):
                bootstrap     = np.random.choice(boot_sample_ids, size=boot_sample_ids.size, replace=True)
                bootstrap_jds = (jd[splits[i]])[bootstrap]
                bootstrap_mag = (mag[splits[i]])[bootstrap]
        
                lsq           = least_squares(return_residuals,
                                              x0 = initial_guess,
                                              args=(bootstrap_jds, bootstrap_mag, clf.coef_, clf.intercept_, period, order))
                boot_store[j] = lsq.x
            
            oc_sd[i] = np.sqrt( np.sum((oc_oc[i]-boot_store)**2)/bootstrap_times )
        
    if bootstrap_times == 0:
        return oc_jd, oc_oc
    else:
        return oc_jd, oc_oc, oc_sd
    
    
# Helper function providing the residuals for the original formulation of the adopted
# O-C shape (LTTE + parabola, with omega and e as variables)
    
def return_residuals_orbit(params, jd, oc, sd=1.):
    return (oc - calc_orbit(params[0], params[1], params[2], params[3], params[4], jd)\
           - params[5] - params[6]*jd - params[7]*jd*jd)/sd

# The function doing the initial O-C fit (LTTE + parabola) using the least_squares
# function from Scipy

def fit_oc1(oc_jd, oc_oc, jd, params, lower_bounds, upper_bounds, plot=True):
    lsq = least_squares(return_residuals_orbit, x0 = params, args=(oc_jd, oc_oc),
                    bounds = (lower_bounds, upper_bounds))

    if plot:
        fig = plt.figure(figsize=(10,6))

        grid      = np.linspace(500,9500,500)
        original  = calc_orbit(params[0], params[1], params[2], params[3], params[4], grid) +\
        params[5] + params[6] * grid + params[7] * grid ** 2

        orb       = calc_orbit(lsq.x[0],  lsq.x[1],  lsq.x[2],  lsq.x[3],  lsq.x[4],  grid) +\
        lsq.x[5]  + lsq.x[6]  * grid + lsq.x[7]  * grid ** 2

        plt.plot(oc_jd, oc_oc, '.')
        plt.plot(grid, orb)
        plt.plot(grid, original, '--', color='grey', alpha=0.5)
        plt.axvline(lsq.x[2], alpha=0.4)
        
        plt.show()

    jd2 = jd - calc_orbit(lsq.x[0] , lsq.x[1], lsq.x[2], lsq.x[3], lsq.x[4], jd) -\
    lsq.x[5] - lsq.x[6] * jd - lsq.x[7] * jd ** 2

    return lsq.x, jd2


# The function calculating the log priors

@jit(nopython=True)
def log_prior(theta, prior_ranges):
    e = np.sqrt(theta[3]**2 + theta[4]**2)

    if e >= 1.0 or theta[0] < prior_ranges[0,0] or theta[0] > prior_ranges[0,1]\
                or theta[1] < prior_ranges[1,0] or theta[1] > prior_ranges[1,1]\
                or theta[2] < prior_ranges[2,0] or theta[2] > prior_ranges[2,1]\
                or theta[5] < prior_ranges[5,0] or theta[5] > prior_ranges[5,1]\
                or theta[6] < prior_ranges[6,0] or theta[6] > prior_ranges[6,1]\
                or theta[7] < prior_ranges[7,0] or theta[7] > prior_ranges[7,1]:
        return -np.inf
    else:
        return 0.0
    
@jit(nopython=True)
def model(theta, x):
    P_orb, T_peri, a_sini, esinomega, ecosomega, a, b, c = theta
    e =         esinomega**2 + ecosomega**2
    omega = np.arctan2(esinomega, ecosomega)
    return calc_orbit(e, P_orb, T_peri, a_sini, omega, x) + a + b*x + c*x*x

# The log likelihood function

@jit(nopython=True)
def log_likelihood(theta, x, y, yerr):
    modelval = model(theta, x)
    sigma2 = yerr ** 2 
    return -0.5 * np.sum((y - modelval) ** 2 / sigma2)

# The log probability function, containing both the likelihood and the prior

@jit(nopython=True) #bad performance!
def log_probability(theta, x, y, yerr, prior_ranges):
    lp = log_prior(theta, prior_ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

# The function doing the MCMC calculations and the calculation of the final
# parameters

def run_mcmc(oc_jd, oc_oc, oc_sd, prior_ranges, initial_state,
              nsteps=6000, discard = 1000, thin = 200, processes=1,
              plot_oc = True, plot_triangle = True):

    nwalkers, ndim = initial_state.shape
    
    if processes==1:
        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        log_probability,
                                        args=(oc_jd, oc_oc, oc_sd, prior_ranges))
        sampler.run_mcmc(initial_state = initial_state, nsteps = nsteps, progress=True);
    else:
        
        with Pool(processes=processes) as pool:
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            log_probability,
                                            args=(oc_jd, oc_oc, oc_sd, prior_ranges),
                                            pool=pool)
            sampler.run_mcmc(initial_state = initial_state, nsteps = nsteps, progress=True);
        
    labels = ["$P_{\mathrm{orb}}$", "$T_{\mathrm{peri}}$", "$a \cdot \sin(i)$",
              "$\sqrt{e} \cdot \sin(\omega)$", "$\sqrt{e} \cdot \cos(\omega)$",
              "$a$", "$b$", "$c$"]
        
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    grid    = np.linspace(0,9500,500)
    grid_oc = np.zeros((flat_samples.shape[0], grid.size))
    for i in tqdm(range(flat_samples.shape[0]), ncols=80):
        grid_oc[i] = model(flat_samples[i,:],grid)
        
    oc_sigmas  = np.percentile(grid_oc, (0.135, 2.275, 15.866, 84.134, 97.725, 99.865),
                               axis=0)
    mcmc_means  = np.mean(flat_samples, axis=0)
    mcmc_sigmas = np.std(flat_samples, axis=0)
    
    if plot_oc:
        fig = plt.figure(figsize=(16,6))
        plt.errorbar(oc_jd, oc_oc, oc_sd, fmt=".k", ms=10, capsize=3)

        fit_mcmc      = model(mcmc_means, grid)
        fit_at_points = model(mcmc_means, oc_jd)
        plt.plot(grid,fit_mcmc,'--')
        plt.ylim(oc_oc.min()-0.01,oc_oc.max()+0.01)
        plt.fill_between(grid, oc_sigmas[2], oc_sigmas[3], facecolor=colorC1, edgecolor=(0,0,0,0))
        plt.fill_between(grid, oc_sigmas[1], oc_sigmas[4], facecolor=colorC1, edgecolor=(0,0,0,0))
        plt.fill_between(grid, oc_sigmas[0], oc_sigmas[5], facecolor=colorC1, edgecolor=(0,0,0,0))

        plt.xlim(grid.min(), grid.max())
        plt.ylim(oc_oc.min()-0.005,oc_oc.max()+0.005)
        plt.show()

    if plot_triangle:
        c = CC().add_chain(flat_samples, parameters=labels, statistics="mean")
        c.configure(sigmas=[0,1,2,3], summary=False, cloud=False, shade=True, shade_gradient=0.99, bar_shade=False,
                sigma2d=True)
        fig = c.plotter.plot()
        plt.show()

    param_means      = np.zeros(8)
    param_sigmas     = np.zeros(8)
    param_means[:3]  = mcmc_means[:3]
    param_means[5:]  = mcmc_means[5:]
    param_sigmas[:3] = mcmc_sigmas[:3]
    param_sigmas[5:] = mcmc_sigmas[5:]
    
    ecc   =            flat_samples[:,3]**2+flat_samples[:,4]**2 
    omega = np.arctan2(flat_samples[:,3],   flat_samples[:,4])

    hist_range   = np.linspace(-np.pi, np.pi, 101)
    hist_mids    = (hist_range[0:-1]+hist_range[1:])/2
    nums_per_bin = np.histogram(omega, hist_range)[0]
    max_bin      = hist_mids[np.argmax(nums_per_bin)]

    omega[omega-max_bin < -np.pi] = omega[omega-max_bin < -np.pi] + 2 * np.pi
    omega[omega-max_bin >  np.pi] = omega[omega-max_bin >  np.pi] - 2 * np.pi
    param_means[3]  = np.mean(ecc)
    param_sigmas[3] = np.std(ecc)
    param_means[4]  = np.mean(omega)
    param_sigmas[4] = np.std(omega)
    
    Ks = calc_K(ecc, flat_samples[:,0], flat_samples[:,1], flat_samples[:,2], omega)
    fm = calc_mass_function(flat_samples[:,2],flat_samples[:,0])
    K  = np.array( (np.mean(Ks), np.std(Ks), np.mean(fm), np.std(fm)) )

    return sampler, fit_mcmc, oc_sigmas, param_means, param_sigmas, fit_at_points, K
