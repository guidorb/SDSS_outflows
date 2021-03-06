####################################################################
##	Program written to fit the profile of a continuum-normalised  ##
##  NaD profile from SDSS DR7 spectra.  						  ##
##	by G. W. Roberts-Borsani						 			  ##
##	17/03/2017										 			  ##
####################################################################

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy
import pymultinest
from . import catalogs as cts


nad_b = 5889.951
nad_r = 5895.924
hei = 	5875.67
c = 3.e5


def NaDprofile(wav, cube, profile=None):
	if (profile=='fixed'):
		u_b = ((wav-nad_b)**2) / (((nad_b*cube[2])/c)**2)
		u_r = ((wav-nad_r)**2) / (((nad_r*cube[2])/c)**2)
		
		tau_b = cube[1]*np.exp(-u_b)
		tau_r = ((cube[1]/2.0)*np.exp(-u_r))
		      
		totfit = 1.0 - cube[0] + cube[0]*np.exp(-tau_b-tau_r)

	elif (profile=='offset'):
		u_b = ((wav-(nad_b+cube[3]))**2) / ((((nad_b+cube[3])*cube[2])/c)**2)
		u_r = ((wav-(nad_r+cube[3]))**2) / ((((nad_r+cube[3])*cube[2])/c)**2)
		
		tau_b = cube[1]*np.exp(-u_b)
		tau_r = ((cube[1]/2.0)*np.exp(-u_r))
		      
		totfit = 1.0 - cube[0] + cube[0]*np.exp(-tau_b-tau_r)

	elif (profile=='double'):
		u_b1 = ((wav-nad_b)**2) / (((nad_b*cube[2])/c)**2)
		u_r1 = ((wav-nad_r)**2) / (((nad_r*cube[2])/c)**2)
		
		tau_b1 = cube[1]*np.exp(-u_b1)
		tau_r1 = ((cube[1]/2.0)*np.exp(-u_r1))
		      
		totfit1 = 1.0 - cube[0] + cube[0]*np.exp(-tau_b1-tau_r1)
		
		
		u_b2 = ((wav-(nad_b+cube[6]))**2) / ((((nad_b+cube[6])*cube[5])/c)**2)
		u_r2 = ((wav-(nad_r+cube[6]))**2) / ((((nad_r+cube[6])*cube[5])/c)**2)
		
		tau_b2 = cube[4]*np.exp(-u_b2)
		tau_r2 = ((cube[4]/2.0)*np.exp(-u_r2))
		      
		totfit2 = 1.0 - cube[3] + cube[3]*np.exp(-tau_b2-tau_r2)
		
		totfit = totfit1 + totfit2 - 1.0

	elif (profile=='triple'):
		u_b1 = ((wav-nad_b)**2) / (((nad_b*cube[2])/c)**2)
		u_r1 = ((wav-nad_r)**2) / (((nad_r*cube[2])/c)**2)
		
		tau_b1 = cube[1]*np.exp(-u_b1)
		tau_r1 = ((cube[1]/2.0)*np.exp(-u_r1))
		      
		totfit1 = 1.0 - cube[0] + cube[0]*np.exp(-tau_b1-tau_r1)
		
		
		u_b2 = ((wav-(nad_b+cube[6]))**2) / ((((nad_b+cube[6])*cube[5])/c)**2)
		u_r2 = ((wav-(nad_r+cube[6]))**2) / ((((nad_r+cube[6])*cube[5])/c)**2)
		
		tau_b2 = cube[4]*np.exp(-u_b2)
		tau_r2 = ((cube[4]/2.0)*np.exp(-u_r2))
		      
		totfit2 = 1.0 - cube[3] + cube[3]*np.exp(-tau_b2-tau_r2)
	
	
		u_b3 = ((wav-(nad_b+cube[10]))**2) / ((((nad_b+cube[10])*cube[9])/c)**2)
		u_r3 = ((wav-(nad_r+cube[10]))**2) / ((((nad_r+cube[10])*cube[9])/c)**2)
		
		tau_b3 = cube[8]*np.exp(-u_b3)
		tau_r3 = ((cube[8]/2.0)*np.exp(-u_r3))
		      
		totfit3 = 1.0 - cube[7] + cube[7]*np.exp(-tau_b3-tau_r3)
		
		totfit = totfit1 + totfit2 + totfit3 - 2.0
	return totfit


def Gaussprofile(wav, cube, profile=None):
	if (profile=='gaussfixed'):
		gauss1 = 1.0 + cube[0]*np.exp(-((wav-nad_b)**2.)/(2.*cube[1]**2.)) 
		gauss2 = 1.0 + cube[2]*np.exp(-((wav-nad_r)**2.)/(2.*cube[1]**2.))
		totgauss = gauss1 + gauss2 - 1.0
	elif (profile=='gaussoffset'):	
		gauss1 = 1.0 + cube[0]*np.exp(-((wav-(nad_b+cube[3]))**2.)/(2.*cube[1]**2.)) 
		gauss2 = 1.0 + cube[2]*np.exp(-((wav-(nad_r+cube[3]))**2.)/(2.*cube[1]**2.))
		totgauss = gauss1 + gauss2 - 1.0
	return totgauss


def NaD_quickprofilefit(wav, normalised, standev, priors, profile=None):
	if (profile=='fixed'):
		def myloglike_fixed(cube, nparams):
			# u_b = ((wav-nad_b)**2) / (((nad_b*cube[2])/c)**2)
			# u_r = ((wav-nad_r)**2) / (((nad_r*cube[2])/c)**2)
			
			# tau_b = cube[1]*np.exp(-u_b)
			# tau_r = ((cube[1]/2.0)*np.exp(-u_r))
			      
			# totfit = 1.0 - cube[0] + cube[0]*np.exp(-tau_b-tau_r)
			totfit = NaDprofile(wav, cube, profile=profile)
		
			loglike = -0.5*np.sum(((normalised - totfit)/standev)**2)
			return loglike
		
		parameters_fixed = [r"C$_{f,r}$", r"$\tau_{0,r}$", r"b_r"]
		n_params_fixed = len(parameters_fixed)
		
		bounds = [ [priors[0], priors[1]],
				   [priors[2], priors[3]],
				   [priors[4], priors[5]]
				]		
		for i in range(750):
			x0 = [ np.random.uniform(priors[0], priors[1]),
					np.random.uniform(priors[2], priors[3]),
					np.random.uniform(priors[4], priors[5])
				 ]
			res = scipy.optimize.minimize(lambda cube: -myloglike_fixed(cube, n_params_fixed), x0 = x0, bounds=bounds )
		
			p_opts = res.x
			myloglike_temp = myloglike_fixed(p_opts, n_params_fixed)
			if (i==0):
				params_fixed = p_opts
				myloglikefixed = myloglike_temp
			if (myloglike_temp > myloglikefixed):
				params_fixed = p_opts
				myloglikefixed = myloglike_temp
		
		totalfit_fixed = NaDprofile(wav, params_fixed, profile=profile)
		return totalfit_fixed, params_fixed, myloglikefixed

	elif (profile=='offset'):
		def myloglike_offset(cube, nparams):
			# u_b = ((wav-(nad_b+cube[3]))**2) / ((((nad_b+cube[3])*cube[2])/c)**2)
			# u_r = ((wav-(nad_r+cube[3]))**2) / ((((nad_r+cube[3])*cube[2])/c)**2)
			
			# tau_b = cube[1]*np.exp(-u_b)
			# tau_r = ((cube[1]/2.0)*np.exp(-u_r))
			      
			# totfit = 1.0 - cube[0] + cube[0]*np.exp(-tau_b-tau_r)
			totfit = NaDprofile(wav, cube, profile=profile)
		
			loglike = -0.5*np.sum(((normalised - totfit)/standev)**2)
			return loglike
	
		parameters_offset = [r"C$_{f,r}$", r"$\tau_{0,r}$", r"b_{r}", r"$\Delta\lambda$"]
		n_params_offset = len(parameters_offset)
		
		bounds = [ [priors[0], priors[1]],
				   [priors[2], priors[3]],
				   [priors[4], priors[5]],
				   [priors[6], priors[7]]
				]		
		for i in range(750):
			x0 = [ np.random.uniform(priors[0], priors[1]),
					np.random.uniform(priors[2], priors[3]),
					np.random.uniform(priors[4], priors[5]),
					np.random.uniform(priors[6], priors[7])
				]
			res = scipy.optimize.minimize(lambda cube: -myloglike_offset(cube, n_params_offset), x0 = x0, bounds=bounds )
		
			p_opts = res.x
			myloglike_temp = myloglike_offset(p_opts, n_params_offset)
			if (i==0):
				params_offset = p_opts
				myloglikeoffset = myloglike_temp
			if (myloglike_temp > myloglikeoffset):
				params_offset = p_opts
				myloglikeoffset = myloglike_temp
		
		totalfit_offset = NaDprofile(wav, params_offset, profile=profile)
		return totalfit_offset, params_offset, myloglikeoffset

	elif (profile=='gaussfixed'):	
		def myloglike_doublegaussfixed(cube, nparams):
			# gauss1 = 1.0 + cube[0]*np.exp(-((wav-nad_b)**2.)/(2.*cube[1]**2.)) 
			# gauss2 = 1.0 + cube[2]*np.exp(-((wav-nad_r)**2.)/(2.*cube[1]**2.))
			# totfit = gauss1 + gauss2 - 1.0
			totfit = Gaussprofile(wav, cube, profile=profile)
	
			loglike = -0.5*np.sum(((normalised - totfit)/standev)**2)
			return loglike
		
		parameters_gauss_fixed = [r"Amp1", r"$\sigma$", r"Amp2"]
		n_params_gauss_fixed = len(parameters_gauss_fixed)
		
		bounds = [ [priors[0], priors[1]],
				   [priors[2], priors[3]],
				   [priors[4], priors[5]],
				]		
		for i in range(750):
			x0 = [ np.random.uniform(priors[0], priors[1]),
					np.random.uniform(priors[2], priors[3]),
					np.random.uniform(priors[4], priors[5]),
				]
			res = scipy.optimize.minimize(lambda cube: -myloglike_doublegaussfixed(cube, n_params_gauss_fixed), x0 = x0, bounds=bounds )
		
			p_opts = res.x
			myloglike_temp = myloglike_doublegaussfixed(p_opts, n_params_gauss_fixed)
			if (i==0):
				params_gauss_fixed = p_opts
				myloglikegauss_fixed = myloglike_temp
			if (myloglike_temp > myloglikegauss_fixed):
				params_gauss_fixed = p_opts
				myloglikegauss_fixed = myloglike_temp
		
		totalfit_gauss_fixed = Gaussprofile(wav, params_gauss_fixed, profile=profile)
		return totalfit_gauss_fixed, params_gauss_fixed, myloglikegauss_fixed

	elif (profile=='gaussoffset'):
		def myloglike_doublegaussoffset(cube, nparams):
			# gauss1 = 1.0 + cube[0]*np.exp(-((wav-(nad_b+cube[3]))**2.)/(2.*cube[1]**2.)) 
			# gauss2 = 1.0 + cube[2]*np.exp(-((wav-(nad_r+cube[3]))**2.)/(2.*cube[1]**2.))
			# totfit = gauss1 + gauss2 - 1.0
			totfit = Gaussprofile(wav, cube, profile=profile)
	
			loglike = -0.5*np.sum(((normalised - totfit)/standev)**2)
			return loglike
		
		parameters_gauss_offset = [r"Amp1", r"$\sigma$", r"Amp2", r"$\Delta\lambda$"]
		n_params_gauss_offset = len(parameters_gauss_offset)
		
		bounds = [ [priors[0], priors[1]],
				   [priors[2], priors[3]],
				   [priors[4], priors[5]],
				   [priors[6], priors[7]]
				]		
		for i in range(750):
			x0 = [ np.random.uniform(priors[0], priors[1]),
					np.random.uniform(priors[2], priors[3]),
					np.random.uniform(priors[4], priors[5]),
					np.random.uniform(priors[6], priors[7])
				]
			res = scipy.optimize.minimize(lambda cube: -myloglike_doublegaussoffset(cube, n_params_gauss_offset), x0 = x0, bounds=bounds )
		
			p_opts = res.x
			myloglike_temp = myloglike_doublegaussoffset(p_opts, n_params_gauss_offset)
			if (i==0):
				params_gauss_offset = p_opts
				myloglikegauss_offset = myloglike_temp
			if (myloglike_temp > myloglikegauss_offset):
				params_gauss_offset = p_opts
				myloglikegauss_offset = myloglike_temp
		
		totalfit_gauss_offset = Gaussprofile(wav, params_gauss_offset, profile=profile)
		return totalfit_gauss_offset, params_gauss_offset, myloglikegauss_offset


def determine_profile(catalog_cont=None):
	# Read in the spectra and continuum models
	stack = catalog_cont['CB08_stack']
	errors = catalog_cont['CB08_stack_err']
	model_init = catalog_cont['CB08_fit']
	wav0 = catalog_cont['CB08_wavelength']

	# Apply a correction factor to normalise continuum to 1.
	norm = stack/model_init
	cok = np.where(((wav0 > 5860.) & (wav0 < 5880.)) | ((wav0 > 5905.) & (wav0 < 5930.)))
	corfactor = np.median(norm[cok])
	model = model_init * corfactor
	
	# Prepare necessary variables
	res = stack/model
	res_err = errors/model
	selection = np.where((wav0 >= 5860.0) & (wav0 <= 5920.0))[0]
	wavf = np.array(wav0[selection[0]:selection[-1]].copy())
	resf = np.array(res[selection[0]:selection[-1]].copy())
	errf = np.array(res_err[selection[0]:selection[-1]].copy())
	modelf = np.array(model[selection[0]:selection[-1]].copy())
	stackf = np.array(stack[selection[0]:selection[-1]].copy())

	# Define the priors for 1-component fitting
	## Single, priors
	Cf_min = -1.0
	Cf_max = 1.0
	tau0_r_min = 1e-10
	tau0_r_max = 20.
	b_r_min = 20.
	b_r_max = 450.
	offset_blue_min = -10.
	offset_blue_max = 0.
	offset_red_min = 0.
	offset_red_max = 10.
	sigma_min = 1e-10
	sigma_max = 5.
	offset_gauss_min = -3.
	offset_gauss_max = 3.

	priors_abs_fixed = [0., Cf_max, tau0_r_min, tau0_r_max, b_r_min, b_r_max]
	priors_abs_offset_blue = [0., Cf_max, tau0_r_min, tau0_r_max, b_r_min, b_r_max, offset_blue_min, offset_blue_max]
	priors_abs_offset_red = [0., Cf_max, tau0_r_min, tau0_r_max, b_r_min, b_r_max, offset_red_min, offset_red_max]
	priors_em_fixed = [Cf_min, 0., tau0_r_min, tau0_r_max, b_r_min, b_r_max]
	priors_em_offset = [Cf_min, 0., tau0_r_min, tau0_r_max, b_r_min, b_r_max, offset_blue_min, offset_red_max]
	
	priors_gauss_abs_fixed = [0.0, -0.1, sigma_min, sigma_max, 0.0, -0.1]
	priors_gauss_abs_offset = [0.0, -0.1, sigma_min, sigma_max, 0.0, -0.1, offset_gauss_min, offset_gauss_max]
	priors_gauss_em_fixed = [0.0, 0.1, sigma_min, sigma_max, 0.0, 0.1]
	priors_gauss_em_offset = [0.0, 0.1, sigma_min, sigma_max, 0.0, 0.1, offset_gauss_min, offset_gauss_max]

	# Determine what kind of profile it is:
	# First, fit a fixed and offset, single, absorption amd emmission component fitting over whole wavelength array
	totalfit_abs_fixed, params_abs_fixed, myloglikeabsfixed = NaD_quickprofilefit(wavf, resf, errf, priors_abs_fixed, profile='fixed')
	totalfit_abs_offset_blue, params_abs_offset_blue, myloglikeabsoffset_blue = NaD_quickprofilefit(wavf, resf, errf, priors_abs_offset_blue, profile='offset')
	totalfit_abs_offset_red, params_abs_offset_red, myloglikeabsoffset_red = NaD_quickprofilefit(wavf, resf, errf, priors_abs_offset_red, profile='offset')
	
	totalfit_em_fixed, params_em_fixed, myloglikeemfixed = NaD_quickprofilefit(wavf, resf, errf, priors_em_fixed, profile='fixed')
	totalfit_em_offset, params_em_offset, myloglikeemoffset = NaD_quickprofilefit(wavf, resf, errf, priors_em_offset, profile='offset')

	# Fit double gaussians, in absorption and in emission, for a p-cygni profile 
	totalfit_abs_gauss_fixed, params_abs_gauss_fixed, myloglikeabsgauss_fixed = NaD_quickprofilefit(wavf, resf, errf, priors_gauss_abs_fixed, profile='gaussfixed')
	totalfit_abs_gauss_offset, params_abs_gauss_offset, myloglikeabsgauss_offset = NaD_quickprofilefit(wavf, resf, errf, priors_gauss_abs_offset, profile='gaussoffset')

	totalfit_em_gauss_fixed, params_em_gauss_fixed, myloglikeemgauss_fixed = NaD_quickprofilefit(wavf, resf, errf, priors_gauss_em_fixed, profile='gaussfixed')
	totalfit_em_gauss_offset, params_em_gauss_offset, myloglikeemgauss_offset = NaD_quickprofilefit(wavf, resf, errf, priors_gauss_em_offset, profile='gaussoffset')

	# Compute likelihood ratios and determine velocity shifts
	kparam_abs_blue = 2.*(myloglikeabsoffset_blue - myloglikeabsfixed)
	kparam_abs_red = 2.*(myloglikeabsoffset_red - myloglikeabsfixed)
	kparam_em = 2.*(myloglikeemoffset - myloglikeemfixed)
	kparam_abs_gauss = 2.*(myloglikeabsgauss_offset - myloglikeabsgauss_fixed)
	kparam_em_gauss = 2.*(myloglikeemgauss_offset - myloglikeemgauss_fixed)

	dV_abs_offset_blue = (c*(params_abs_offset_blue[-1]/nad_r))
	dV_abs_offset_red = (c*(params_abs_offset_red[-1]/nad_r))
	dV_em_offset = (c*(params_em_offset[-1]/nad_r))
	dV_abs_offset_gauss = (c*(params_abs_gauss_offset[-1]/nad_r))
	dV_em_offset_gauss = (c*(params_em_gauss_offset[-1]/nad_r))

	# Determine the type of profile:
	if ((kparam_abs_blue > 0.) & (kparam_abs_red > 0.)) & (kparam_em_gauss == 0.):
		profile = 'absorption'
	elif ((kparam_em > 0.) | (kparam_em_gauss > 0.)) & (kparam_abs_blue == 0.) & (kparam_abs_red == 0.) & (kparam_abs_gauss == 0.):
		profile = 'emission'
	elif (kparam_abs_gauss > 0.) & (kparam_em_gauss > 0.):
		profile = 'pcygni'
	else:
		profile = 'unknown'

	parameters = {'loglike_abs_fixed':myloglikeabsfixed, 
					'loglike_abs_offset_blue':myloglikeabsoffset_blue, 
					'loglike_abs_offset_red':myloglikeabsoffset_red, 
					'loglike_em_fixed':myloglikeemfixed,
					'loglike_em_offset':myloglikeemoffset,
					'loglike_abs_gauss_fixed':myloglikeabsgauss_fixed,
					'loglike_abs_gauss_offset':myloglikeabsgauss_offset,
					'loglike_em_gauss_fixed':myloglikeemgauss_fixed,
					'loglike_em_gauss_offset':myloglikeemgauss_offset,

					'kparam_abs_blue':kparam_abs_blue,
					'kparam_abs_red':kparam_abs_red,
					'kparam_em':kparam_em,
					'kparam_abs_gauss':kparam_abs_gauss,
					'kparam_em_gauss':kparam_em_gauss,

					'dV_abs_offset_blue':dV_abs_offset_blue,
					'dV_abs_offset_red':dV_abs_offset_red,
					'dV_em_offset':dV_em_offset,
					'dV_abs_offset_gauss':dV_abs_offset_gauss,
					'dV_em_offset_gauss':dV_em_offset_gauss}

	fits = {'totalfit_abs_fixed':totalfit_abs_fixed,
			'totalfit_abs_offset_blue':totalfit_abs_offset_blue,
			'totalfit_abs_offset_red':totalfit_abs_offset_red,
			'totalfit_em_fixed':totalfit_em_fixed,
			'totalfit_em_offset':totalfit_em_offset,
			'totalfit_abs_gauss_fixed':totalfit_abs_gauss_fixed,
			'totalfit_abs_gauss_offset':totalfit_abs_gauss_offset,
			'totalfit_em_gauss_fixed':totalfit_em_gauss_fixed,
			'totalfit_em_gauss_offset':totalfit_em_gauss_offset}

	return profile, parameters, fits


def detect(profile=None, parameters=None):
	kparam_lim = 75.
	deltaV_lim = 20.
	if (profile == 'absorption'):
		if ((parameters['kparam_abs_blue'] >= kparam_lim) & (abs(parameters['dV_abs_offset_blue']) >= deltaV_lim)) & ((parameters['kparam_abs_red'] > kparam_lim) & (abs(parameters['dV_abs_offset_red']) >= deltaV_lim)):
			detection = True
			typ = 'inflow+outflow'
		elif ((parameters['kparam_abs_blue'] >= kparam_lim) & (abs(parameters['dV_abs_offset_blue']) >= deltaV_lim)) & ((parameters['kparam_abs_red'] < kparam_lim) | (abs(parameters['dV_abs_offset_red']) < deltaV_lim)):
			detection = True
			typ = 'outflow'
		elif ((parameters['kparam_abs_red'] >= kparam_lim) & (abs(parameters['dV_abs_offset_red']) >= deltaV_lim)) & ((parameters['kparam_abs_blue'] < kparam_lim) | (abs(parameters['dV_abs_offset_blue']) < deltaV_lim)):
			detection = True
			typ = 'inflow'
		elif ((parameters['kparam_abs_red'] < kparam_lim) | (abs(parameters['dV_abs_offset_red']) < deltaV_lim)) & ((parameters['kparam_abs_blue'] < kparam_lim) | (abs(parameters['dV_abs_offset_blue']) < deltaV_lim)):
			detection = False
			typ = 'non-detection'

	if (profile == 'emission'):
		if ((parameters['kparam_em'] >= kparam_lim) & (abs(parameters['dV_em_offset']) >= deltaV_lim) & (parameters['dV_em_offset'] > 0)):
			detection = True
			typ = 'inflow'
		elif ((parameters['kparam_em'] >= kparam_lim) & (abs(parameters['dV_em_offset']) >= deltaV_lim) & (parameters['dV_em_offset'] < 0)):
			detection = True
			typ = 'outflow'
		else:
			detection = False
			typ = 'non-detection'

	if (profile == 'pcygni'):
		if (parameters['kparam_abs_gauss'] >= kparam_lim) & (parameters['kparam_em_gauss'] >= kparam_lim):
			detection = True
			typ = 'outflow'
		else:
			detection = False
			typ = 'non-detection'

	if (profile == 'unknown'):
		detection = False
		typ = 'non-detection'

	return detection, typ


def fit_PyMultinest(priors, wavf, resf, standev, name, samp, galtype):
	if ((len(priors)/2.) == 3.):
		def myprior_single(cube, ndim, nparams):
			cube[0] = cube[0] * (priors[1] - priors[0]) + priors[0]
			cube[1] = cube[1] * (priors[3] - priors[2]) + priors[2]
			cube[2] = cube[2] * (priors[5] - priors[4]) + priors[4]
		
		def myloglike_single(cube, ndim, nparams):
			# u_b1 = ((wavf-nad_b)**2) / (((nad_b*cube[2])/c)**2)
			# u_r1 = ((wavf-nad_r)**2) / (((nad_r*cube[2])/c)**2)
			
			# tau_b1 = cube[1]*np.exp(-u_b1)
			# tau_r1 = ((cube[1]/2.0)*np.exp(-u_r1))
			      
			# totfit = 1.0 - cube[0] + cube[0]*np.exp(-tau_b1-tau_r1)
			totfit = NaDprofile(wavf, cube, profile='fixed')

			loglike = -0.5*np.sum(((resf - totfit)/standev)**2)
			return loglike
		
		n_params = int(len(priors)/2.)
		pymultinest.run(myloglike_single, myprior_single, n_params, importance_nested_sampling=False, resume=False, verbose=False, sampling_efficiency='model', n_live_points=750, outputfiles_basename='/Users/guidorb/GoogleDrive/SDSS/Bayesian/Bayesian_fit1-'+name+'_'+samp+'_'+galtype)
		analysis = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='/Users/guidorb/GoogleDrive/SDSS/Bayesian/Bayesian_fit1-'+name+'_'+samp+'_'+galtype)
		data = analysis.get_mode_stats()
		best_params = data['modes'][0]['maximum a posterior']
		best_params_err = data['modes'][0]['sigma']
		return best_params, best_params_err

	elif ((len(priors)/2.) == 7.):
		def myprior_double(cube, ndim, nparams):
			cube[0] = cube[0] * (priors[1] - priors[0]) + priors[0]
			cube[1] = cube[1] * (priors[3] - priors[2]) + priors[2]
			cube[2] = cube[2] * (priors[5] - priors[4]) + priors[4]
		
			cube[3] = cube[3] * (priors[7] - priors[6]) + priors[6]
			cube[4] = cube[4] * (priors[9] - priors[8]) + priors[8]
			cube[5] = cube[5] * (priors[11] - priors[10]) + priors[10]
			cube[6] = cube[6] * (priors[13] - priors[12]) + priors[12]
		
		def myloglike_double(cube, ndim, nparams):
			# u_b1 = ((wavf-nad_b)**2) / (((nad_b*cube[2])/c)**2)
			# u_r1 = ((wavf-nad_r)**2) / (((nad_r*cube[2])/c)**2)
			
			# tau_b1 = cube[1]*np.exp(-u_b1)
			# tau_r1 = ((cube[1]/2.0)*np.exp(-u_r1))
			      
			# totfit1 = 1.0 - cube[0] + cube[0]*np.exp(-tau_b1-tau_r1)
			
			
			# u_b2 = ((wavf-(nad_b+cube[6]))**2) / ((((nad_b+cube[6])*cube[5])/c)**2)
			# u_r2 = ((wavf-(nad_r+cube[6]))**2) / ((((nad_r+cube[6])*cube[5])/c)**2)
			
			# tau_b2 = cube[4]*np.exp(-u_b2)
			# tau_r2 = ((cube[4]/2.0)*np.exp(-u_r2))
			      
			# totfit2 = 1.0 - cube[3] + cube[3]*np.exp(-tau_b2-tau_r2)
			
			# totfit = totfit1 + totfit2 - 1.0
			totfit = NaDprofile(wavf, cube, profile='double')
			loglike = -0.5*np.sum(((resf - totfit)/standev)**2)
			return loglike
		
		n_params = int(len(priors)/2.)
		pymultinest.run(myloglike_double, myprior_double, n_params, importance_nested_sampling=False, resume=False, verbose=False, sampling_efficiency='model', n_live_points=750, outputfiles_basename='/Users/guidorb/GoogleDrive/SDSS/Bayesian/Bayesian_fit1-'+name+'_'+samp+'_'+galtype)
		analysis = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='/Users/guidorb/GoogleDrive/SDSS/Bayesian/Bayesian_fit1-'+name+'_'+samp+'_'+galtype)
		data = analysis.get_mode_stats()
		best_params = data['modes'][0]['maximum a posterior']
		best_params_err = data['modes'][0]['sigma']
		return best_params, best_params_err

	elif ((len(priors)/2.) == 8.):
		def myprior_triple(cube, ndim, nparams):
			cube[0] = cube[0] * (priors[1] - priors[0]) + priors[0]
			cube[1] = cube[1] * (priors[3] - priors[2]) + priors[2]
			cube[2] = cube[2] * (priors[5] - priors[4]) + priors[4]
		
			cube[3] = cube[3] * (priors[7] - priors[6]) + priors[6]
			cube[4] = cube[4] * (priors[9] - priors[8]) + priors[8]
			cube[5] = cube[5] * (priors[11] - priors[10]) + priors[10]
			cube[6] = cube[6] * (priors[13] - priors[12]) + priors[12]

			cube[7] = cube[7] * (priors[7] - priors[6]) + priors[6]
			cube[8] = cube[8] * (priors[9] - priors[8]) + priors[8]
			cube[9] = cube[9] * (priors[11] - priors[10]) + priors[10]
			cube[10] = cube[10] * (priors[15] - priors[14]) + priors[14]
		
		def myloglike_triple(cube, ndim, nparams):
			# u_b1 = ((wavf-nad_b)**2) / (((nad_b*cube[2])/c)**2)
			# u_r1 = ((wavf-nad_r)**2) / (((nad_r*cube[2])/c)**2)
			
			# tau_b1 = cube[1]*np.exp(-u_b1)
			# tau_r1 = ((cube[1]/2.0)*np.exp(-u_r1))
			      
			# totfit1 = 1.0 - cube[0] + cube[0]*np.exp(-tau_b1-tau_r1)
			
			
			# u_b2 = ((wavf-(nad_b+cube[6]))**2) / ((((nad_b+cube[6])*cube[5])/c)**2)
			# u_r2 = ((wavf-(nad_r+cube[6]))**2) / ((((nad_r+cube[6])*cube[5])/c)**2)
			
			# tau_b2 = cube[4]*np.exp(-u_b2)
			# tau_r2 = ((cube[4]/2.0)*np.exp(-u_r2))
			      
			# totfit2 = 1.0 - cube[3] + cube[3]*np.exp(-tau_b2-tau_r2)


			# u_b3 = ((wavf-(nad_b+cube[10]))**2) / ((((nad_b+cube[10])*cube[9])/c)**2)
			# u_r3 = ((wavf-(nad_r+cube[10]))**2) / ((((nad_r+cube[10])*cube[9])/c)**2)
			
			# tau_b3 = cube[8]*np.exp(-u_b3)
			# tau_r3 = ((cube[8]/2.0)*np.exp(-u_r3))
			      
			# totfit3 = 1.0 - cube[7] + cube[7]*np.exp(-tau_b3-tau_r3)
			
			# totfit = totfit1 + totfit2 + totfit3 - 2.0
			totfit = NaDprofile(wavf, cube, profile='triple')
			loglike = -0.5*np.sum(((resf - totfit)/standev)**2)
			return loglike
		
		n_params = int((len(priors)/2.)+3)
		pymultinest.run(myloglike_triple, myprior_triple, n_params, importance_nested_sampling=False, resume=False, verbose=False, sampling_efficiency='model', n_live_points=750, outputfiles_basename='/Users/guidorb/GoogleDrive/SDSS/Bayesian/Bayesian_fit1-'+name+'_'+samp+'_'+galtype)
		analysis = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='/Users/guidorb/GoogleDrive/SDSS/Bayesian/Bayesian_fit1-'+name+'_'+samp+'_'+galtype)
		data = analysis.get_mode_stats()
		best_params = data['modes'][0]['maximum a posterior']
		best_params_err = data['modes'][0]['sigma']
		return best_params, best_params_err



def nadfit_process(name, samp, galtype, save=False, sn_limit=None, adaptive=False, dimension=None):
		catalog = cts.call_savedcatalog(name, samp, galtype, 'fitcont', sn_limit=sn_limit, adaptive=adaptive, dimension=dimension)
		catalog_stack = cts.call_savedcatalog(name, samp, galtype, 'stack', sn_limit=sn_limit, adaptive=adaptive, dimension=dimension)
		
		catalog_new = {}

		for binning in catalog_stack['bins']:
			if 'CB08_stack' in catalog[binning]:
				print(binning + '/' + str(np.size(catalog_stack['bins'])) + '      NaD fitting...')

				profiletype, parameters, fits = determine_profile(catalog_cont=catalog[binning])
				det, typ = detect(profile=profiletype, parameters=parameters)

				# Prepare arrays to be fit and apply a correction factor to normalise continuum to 1.
				wav0 = catalog[binning]['CB08_wavelength']
				stack = catalog[binning]['CB08_stack']
				errors = catalog[binning]['CB08_stack_err']
				model_init = catalog[binning]['CB08_fit']
				
				norm = stack/model_init
				cok = np.where(((wav0 > 5860.) & (wav0 < 5880.)) | ((wav0 > 5905.) & (wav0 < 5930.)))
				corfactor = np.median(norm[cok])
				model = model_init * corfactor
		
				res = stack/model
				res_err = errors/model
		
				selection = np.where((wav0 >= 5860.0) & (wav0 <= 5920.0))[0]
				wavf = np.array(wav0[selection[0]:selection[-1]].copy())
				resf = np.array(res[selection[0]:selection[-1]].copy())
				errf = np.array(res_err[selection[0]:selection[-1]].copy())
				modelf = np.array(model[selection[0]:selection[-1]].copy())
				stackf = np.array(stack[selection[0]:selection[-1]].copy())	
				He_width = np.sqrt(2.)*(catalog[binning]['CB08_params'][81]/nad_r)*c

				if (profiletype=='absorption') & (det==True) & ((typ=='outflow') | (typ=='inflow')):
					## Single, priors
					Cf_min = 0
					Cf_max = 1.
					tau0_r_min = 1e-10
					tau0_r_max = 20.
					b_r_min = 20.
					b_r_max = 450.
					offset_blue_min = -10.
					offset_blue_max = 0.
					offset_red_min = 0.
					offset_red_max = 10.
					sigma_min = 1e-10
					sigma_max = 10.
					offset_gauss_min = -3.
					offset_gauss_max = 3.
	
					priors_single = [Cf_min, Cf_max, tau0_r_min, tau0_r_max, b_r_min, b_r_max]
					params_single, params_single_err = fit_PyMultinest(priors_single, wavf, resf, errf, name, samp, galtype)
	
					## Double, priors
					Cf_min = 0.
					Cf_max = 0.5
					tau0_min_sys = 1e-10
					tau0_max_sys = 10.0
					tau0_min_off = 1e-10
					tau0_max_off = 10.0
					b_min_sys = 20.
					b_max_sys = 450.
					b_min_off = 20.
					b_max_off = 200.
					if (typ=='outflow'):
						offset_min = -15.
						offset_max = 0.
					if (typ=='inflow'):
						offset_min = 0.
						offset_max = 15.
	
					priors_double = [Cf_min, 1.0, tau0_min_sys, tau0_max_sys, b_min_sys, b_max_sys, Cf_min, Cf_max, tau0_min_off, tau0_max_off, b_min_off, b_max_off, offset_min, offset_max]
					params_double, params_double_err = fit_PyMultinest(priors_double, wavf, resf, errf, name, samp, galtype)

					totalfit_single = NaDprofile(wavf, params_single, profile='fixed')
					totalfit_double = NaDprofile(wavf, params_double, profile='double')
					systemicfit_double = NaDprofile(wavf, params_double[0:3], profile='fixed')
					outflowfit_double = NaDprofile(wavf, params_double[3:], profile='offset')
	
					nadfitcat = {'wav':wavf, 
							'spec':stackf, 
							'res':resf, 
							'errs':errf,
	
							'totalfit_single':totalfit_single,
							'params_single':params_single,
							'params_single_err':params_single_err,
	
							'totalfit_double':totalfit_double,
							'systemicfit_double':systemicfit_double,
							'outflowfit_double':outflowfit_double,
							'params_double':params_double,
							'params_double_err':params_double_err,
							
							'He_b':He_width}

				elif (profiletype=='emission') & (det==True) & ((typ=='outflow') | (typ=='inflow')):
					## Single, priors
					Cf_min = -1.
					Cf_max = 0
					tau0_r_min = 1e-10
					tau0_r_max = 20.
					b_r_min = 20.
					b_r_max = 450.
					offset_blue_min = -10.
					offset_blue_max = 0.
					offset_red_min = 0.
					offset_red_max = 10.
					sigma_min = 1e-10
					sigma_max = 10.
					offset_gauss_min = -3.
					offset_gauss_max = 3.
	
					priors_single = [Cf_min, Cf_max, tau0_r_min, tau0_r_max, b_r_min, b_r_max]
					params_single, params_single_err = fit_PyMultinest(priors_single, wavf, resf, errf, name, samp, galtype)
	
					## Double, priors
					Cf_min = -0.5
					Cf_max = 0.
					tau0_min_sys = 1e-10
					tau0_max_sys = 10.0
					tau0_min_off = 1e-10
					tau0_max_off = 10.0
					b_min_sys = 20.
					b_max_sys = 200.
					b_min_off = 20.
					b_max_off = 200.
					if (det=='outflow'):
						offset_min = -5.
						offset_max = 0.
					if (det=='inflow'):
						offset_min = 0.
						offset_max = 5.
	
					priors_double = [-1., Cf_max, tau0_min_sys, tau0_max_sys, b_min_sys, b_max_sys, Cf_min, Cf_max, tau0_min_off, tau0_max_off, b_min_off, b_max_off, offset_min, offset_max]
					params_double, params_double_err = fit_PyMultinest(priors_double, wavf, resf, errf, name, samp, galtype)

					totalfit_single = NaDprofile(wavf, params_single, profile='fixed')
					totalfit_double = NaDprofile(wavf, params_double, profile='double')
					systemicfit_double = NaDprofile(wavf, params_double[0:3], profile='fixed')
					outflowfit_double = NaDprofile(wavf, params_double[3:], profile='offset')
	
					nadfitcat = {'wav':wavf, 
							'spec':stackf, 
							'res':resf, 
							'errs':errf,
	
							'totalfit_single':totalfit_single,
							'params_single':params_single,
							'params_single_err':params_single_err,
	
							'totalfit_double':totalfit_double,
							'systemicfit_double':systemicfit_double,
							'outflowfit_double':outflowfit_double,
							'params_double':params_double,
							'params_double_err':params_double_err,
							
							'He_b':He_width}

				elif (profiletype=='absorption') & (det==False):
					## Single, priors
					Cf_min = 0.
					Cf_max = 1.
					tau0_r_min = 1e-10
					tau0_r_max = 20.
					b_r_min = 20.
					b_r_max = 450.
					offset_blue_min = -10.
					offset_blue_max = 0.
					offset_red_min = 0.
					offset_red_max = 10.
					sigma_min = 1e-10
					sigma_max = 10.
					offset_gauss_min = -3.
					offset_gauss_max = 3.
	
					priors_single = [Cf_min, Cf_max, tau0_r_min, tau0_r_max, b_r_min, b_r_max]
					params_single, params_single_err = fit_PyMultinest(priors_single, wavf, resf, errf, name, samp, galtype)
					
					totalfit_single = NaDprofile(wavf, params_single, profile='fixed')

					nadfitcat = {'wav':wavf, 
							'spec':stackf, 
							'res':resf, 
							'errs':errf,
	
							'totalfit_single':totalfit_single,
							'params_single':params_single,
							'params_single_err':params_single_err,
	
							'He_b':He_width}

				elif (profiletype=='emission') & (det==False):
					## Single, priors
					Cf_min = -1.
					Cf_max = 0
					tau0_r_min = 1e-10
					tau0_r_max = 20.
					b_r_min = 20.
					b_r_max = 450.
					offset_blue_min = -10.
					offset_blue_max = 0.
					offset_red_min = 0.
					offset_red_max = 10.
					sigma_min = 1e-10
					sigma_max = 10.
					offset_gauss_min = -3.
					offset_gauss_max = 3.
	
					priors_single = [Cf_min, Cf_max, tau0_r_min, tau0_r_max, b_r_min, b_r_max]
					params_single, params_single_err = fit_PyMultinest(priors_single, wavf, resf, errf, name, samp, galtype)
					
					totalfit_single = NaDprofile(wavf, params_single, profile='fixed')

					nadfitcat = {'wav':wavf, 
							'spec':stackf, 
							'res':resf, 
							'errs':errf,
	
							'totalfit_single':totalfit_single,
							'params_single':params_single,
							'params_single_err':params_single_err,
	
							'He_b':He_width}

				elif (profiletype=='pcygni'):
					continue

				elif (profiletype=='unknown'):
					## Single, priors
					Cf_min = -1.
					Cf_max = 1.
					tau0_r_min = 1e-10
					tau0_r_max = 20.
					b_r_min = 20.
					b_r_max = 450.
					offset_blue_min = -10.
					offset_blue_max = 0.
					offset_red_min = 0.
					offset_red_max = 10.
					sigma_min = 1e-10
					sigma_max = 10.
					offset_gauss_min = -3.
					offset_gauss_max = 3.
	
					priors_single = [Cf_min, Cf_max, tau0_r_min, tau0_r_max, b_r_min, b_r_max]
					params_single, params_single_err = fit_PyMultinest(priors_single, wavf, resf, errf, name, samp, galtype)
					
					totalfit_single = NaDprofile(wavf, params_single, profile='fixed')

					nadfitcat = {'wav':wavf, 
							'spec':stackf, 
							'res':resf, 
							'errs':errf,
	
							'totalfit_single':totalfit_single,
							'params_single':params_single,
							'params_single_err':params_single_err,
	
							'He_b':He_width}

				catalog_new[binning] = nadfitcat
		if (save==True):
			print('Saving...')
			if (sn_limit != None) & (adaptive == True):
				pickle.dump(catalog_new, open('/Users/guidorb/GoogleDrive/SDSS/stacked/SDSSstackcatalog_nadfit_'+name+'_'+dimension+'Dadaptive_SN'+str(sn_limit)+'_'+samp+'_'+galtype+'.p', 'wb'))
			else:
				pickle.dump(catalog_new, open('/Users/guidorb/GoogleDrive/SDSS/stacked/SDSSstackcatalog_nadfit_'+name+'_'+dimension+'D_'+samp+'_'+galtype+'.p', 'wb'))
			return catalog_new
		else:
			return catalog_new



class Fitting():
	def __init__(self, name, samp, galtype, save=False, sn_limit=None, adaptive=False, dimension=None):
		nadcatalogall = nadfit_process(name, samp, galtype, save=save, sn_limit=sn_limit, adaptive=adaptive, dimension=dimension)
		self.catalog = nadcatalogall

