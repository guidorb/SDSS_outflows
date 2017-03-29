#######################################################
##	Program written to fit the continuum of stacked  ##
##  SDSS DR7 spectra.							     ##
##	by G. W. Roberts-Borsani						 ##
##	17/03/2017										 ##
#######################################################

from . import catalogs as cts
import os
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def fit_continuum(name, samp, galtype, sn_limit=None, adaptive=None, dimension='2', save=False):
	def battisti(wav, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, 
					A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, 
					A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, 
					A31, A32, A33, A34, A35, A36, A37, A38, A39, A40, 
					A41, A42, A43, A44, A45, A46, A47, A48, A49, A50, 
					A51, A52, A53, A54, A55, A56, A57, A58, A59, A60, 
					A61, A62, A63, A64, A65, A66, A67, A68, A69, A70, 
					A71, A72, A73, A74, A75, A76, A77, A78, A79, A80, 
					A81, A82):
		R_v = 4.05
		wav_mu = wav/(1e4)
		x = 1./wav_mu
		
		Qfit = np.polyval([0.0145, -0.261, 1.803, -2.488], x)
		klam = 2.396 * Qfit + R_v
		
		yfit = (
				(A0 * new_temps[0]) * 10.0**(-0.4*((klam*A40)/R_v)) + 
				(A1 * new_temps[1]) * 10.0**(-0.4*((klam*A41)/R_v)) + 
				(A2 * new_temps[2]) * 10.0**(-0.4*((klam*A42)/R_v)) + 
				(A3 * new_temps[3]) * 10.0**(-0.4*((klam*A43)/R_v)) + 
				(A4 * new_temps[4]) * 10.0**(-0.4*((klam*A44)/R_v)) + 
				(A5 * new_temps[5]) * 10.0**(-0.4*((klam*A45)/R_v)) + 
				(A6 * new_temps[6]) * 10.0**(-0.4*((klam*A46)/R_v)) + 
				(A7 * new_temps[7]) * 10.0**(-0.4*((klam*A47)/R_v)) + 
				(A8 * new_temps[8]) * 10.0**(-0.4*((klam*A48)/R_v)) + 
				(A9 * new_temps[9]) * 10.0**(-0.4*((klam*A49)/R_v)) + 
				(A10 * new_temps[10]) * 10.0**(-0.4*((klam*A50)/R_v)) + 
				(A11 * new_temps[11]) * 10.0**(-0.4*((klam*A51)/R_v)) + 
				(A12 * new_temps[12]) * 10.0**(-0.4*((klam*A52)/R_v)) + 
				(A13 * new_temps[13]) * 10.0**(-0.4*((klam*A53)/R_v)) + 
				(A14 * new_temps[14]) * 10.0**(-0.4*((klam*A54)/R_v)) + 
				(A15 * new_temps[15]) * 10.0**(-0.4*((klam*A55)/R_v)) + 
				(A16 * new_temps[16]) * 10.0**(-0.4*((klam*A56)/R_v)) + 
				(A17 * new_temps[17]) * 10.0**(-0.4*((klam*A57)/R_v)) + 
				(A18 * new_temps[18]) * 10.0**(-0.4*((klam*A58)/R_v)) + 
				(A19 * new_temps[19]) * 10.0**(-0.4*((klam*A59)/R_v)) + 
				(A20 * new_temps[20]) * 10.0**(-0.4*((klam*A60)/R_v)) + 
				(A21 * new_temps[21]) * 10.0**(-0.4*((klam*A61)/R_v)) + 
				(A22 * new_temps[22]) * 10.0**(-0.4*((klam*A62)/R_v)) + 
				(A23 * new_temps[23]) * 10.0**(-0.4*((klam*A63)/R_v)) + 
				(A24 * new_temps[24]) * 10.0**(-0.4*((klam*A64)/R_v)) + 
				(A25 * new_temps[25]) * 10.0**(-0.4*((klam*A65)/R_v)) + 
				(A26 * new_temps[26]) * 10.0**(-0.4*((klam*A66)/R_v)) + 
				(A27 * new_temps[27]) * 10.0**(-0.4*((klam*A67)/R_v)) + 
				(A28 * new_temps[28]) * 10.0**(-0.4*((klam*A68)/R_v)) + 
				(A29 * new_temps[29]) * 10.0**(-0.4*((klam*A69)/R_v)) + 
				(A30 * new_temps[30]) * 10.0**(-0.4*((klam*A70)/R_v)) + 
				(A31 * new_temps[31]) * 10.0**(-0.4*((klam*A71)/R_v)) + 
				(A32 * new_temps[32]) * 10.0**(-0.4*((klam*A72)/R_v)) + 
				(A33 * new_temps[33]) * 10.0**(-0.4*((klam*A73)/R_v)) + 
				(A34 * new_temps[34]) * 10.0**(-0.4*((klam*A74)/R_v)) + 
				(A35 * new_temps[35]) * 10.0**(-0.4*((klam*A75)/R_v)) + 
				(A36 * new_temps[36]) * 10.0**(-0.4*((klam*A76)/R_v)) + 
				(A37 * new_temps[37]) * 10.0**(-0.4*((klam*A77)/R_v)) + 
				(A38 * new_temps[38]) * 10.0**(-0.4*((klam*A78)/R_v)) + 
				(A39 * new_temps[39]) * 10.0**(-0.4*((klam*A79)/R_v))
				)      
		# idx_low = np.argmin(abs(wav - 5865.))
		idx_low = np.argmin(abs(wav - 5860.))
		# idx_high = np.argmin(abs(wav - 5879.0))
		idx_high = np.argmin(abs(wav - 5875.67))
		gaussf = A82*np.ones(len(wav[idx_low:idx_high])) + A80*np.exp(-((wav[idx_low:idx_high]-5875.67)**2.)/(2.*A81*A81))
		yfit[idx_low:idx_high] = yfit[idx_low:idx_high] + gaussf
		return yfit

	def battisti_full(wav, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, 
						A11, A12, A13, A14, A15, A16, A17, A18, A19, A20, 
						A21, A22, A23, A24, A25, A26, A27, A28, A29, A30, 
						A31, A32, A33, A34, A35, A36, A37, A38, A39, A40, 
						A41, A42, A43, A44, A45, A46, A47, A48, A49, A50, 
						A51, A52, A53, A54, A55, A56, A57, A58, A59, A60, 
						A61, A62, A63, A64, A65, A66, A67, A68, A69, A70, 
						A71, A72, A73, A74, A75, A76, A77, A78, A79, A80, 
						A81, A82):
		R_v = 4.05
		wav_mu = wav/(1e4)
		x = 1./wav_mu
		
		Qfit = np.polyval([0.0145, -0.261, 1.803, -2.488], x)
		klam = 2.396 * Qfit + R_v
		
		yfit = (
				(A0 * new_temps[0]) * 10.0**(-0.4*((klam*A40)/R_v)) + 
				(A1 * new_temps[1]) * 10.0**(-0.4*((klam*A41)/R_v)) + 
				(A2 * new_temps[2]) * 10.0**(-0.4*((klam*A42)/R_v)) + 
				(A3 * new_temps[3]) * 10.0**(-0.4*((klam*A43)/R_v)) + 
				(A4 * new_temps[4]) * 10.0**(-0.4*((klam*A44)/R_v)) + 
				(A5 * new_temps[5]) * 10.0**(-0.4*((klam*A45)/R_v)) + 
				(A6 * new_temps[6]) * 10.0**(-0.4*((klam*A46)/R_v)) + 
				(A7 * new_temps[7]) * 10.0**(-0.4*((klam*A47)/R_v)) + 
				(A8 * new_temps[8]) * 10.0**(-0.4*((klam*A48)/R_v)) + 
				(A9 * new_temps[9]) * 10.0**(-0.4*((klam*A49)/R_v)) + 
				(A10 * new_temps[10]) * 10.0**(-0.4*((klam*A50)/R_v)) + 
				(A11 * new_temps[11]) * 10.0**(-0.4*((klam*A51)/R_v)) + 
				(A12 * new_temps[12]) * 10.0**(-0.4*((klam*A52)/R_v)) + 
				(A13 * new_temps[13]) * 10.0**(-0.4*((klam*A53)/R_v)) + 
				(A14 * new_temps[14]) * 10.0**(-0.4*((klam*A54)/R_v)) + 
				(A15 * new_temps[15]) * 10.0**(-0.4*((klam*A55)/R_v)) + 
				(A16 * new_temps[16]) * 10.0**(-0.4*((klam*A56)/R_v)) + 
				(A17 * new_temps[17]) * 10.0**(-0.4*((klam*A57)/R_v)) + 
				(A18 * new_temps[18]) * 10.0**(-0.4*((klam*A58)/R_v)) + 
				(A19 * new_temps[19]) * 10.0**(-0.4*((klam*A59)/R_v)) + 
				(A20 * new_temps[20]) * 10.0**(-0.4*((klam*A60)/R_v)) + 
				(A21 * new_temps[21]) * 10.0**(-0.4*((klam*A61)/R_v)) + 
				(A22 * new_temps[22]) * 10.0**(-0.4*((klam*A62)/R_v)) + 
				(A23 * new_temps[23]) * 10.0**(-0.4*((klam*A63)/R_v)) + 
				(A24 * new_temps[24]) * 10.0**(-0.4*((klam*A64)/R_v)) + 
				(A25 * new_temps[25]) * 10.0**(-0.4*((klam*A65)/R_v)) + 
				(A26 * new_temps[26]) * 10.0**(-0.4*((klam*A66)/R_v)) + 
				(A27 * new_temps[27]) * 10.0**(-0.4*((klam*A67)/R_v)) + 
				(A28 * new_temps[28]) * 10.0**(-0.4*((klam*A68)/R_v)) + 
				(A29 * new_temps[29]) * 10.0**(-0.4*((klam*A69)/R_v)) + 
				(A30 * new_temps[30]) * 10.0**(-0.4*((klam*A70)/R_v)) + 
				(A31 * new_temps[31]) * 10.0**(-0.4*((klam*A71)/R_v)) + 
				(A32 * new_temps[32]) * 10.0**(-0.4*((klam*A72)/R_v)) + 
				(A33 * new_temps[33]) * 10.0**(-0.4*((klam*A73)/R_v)) + 
				(A34 * new_temps[34]) * 10.0**(-0.4*((klam*A74)/R_v)) + 
				(A35 * new_temps[35]) * 10.0**(-0.4*((klam*A75)/R_v)) + 
				(A36 * new_temps[36]) * 10.0**(-0.4*((klam*A76)/R_v)) + 
				(A37 * new_temps[37]) * 10.0**(-0.4*((klam*A77)/R_v)) + 
				(A38 * new_temps[38]) * 10.0**(-0.4*((klam*A78)/R_v)) + 
				(A39 * new_temps[39]) * 10.0**(-0.4*((klam*A79)/R_v))
				)
		gaussf = A82*np.ones(len(wav)) + A80*np.exp(-((wav-5875.67)**2.)/(2.*A81*A81))
		yfit = yfit + gaussf
		return yfit

	catalog = cts.call_savedcatalog(name, samp, galtype, 'stack', sn_limit=sn_limit, adaptive=adaptive, dimension=dimension)

	data_path = '/Users/guidorb/GoogleDrive/SDSS/'
	models_dir_cb08 = '/Users/guidorb/AstroSoftware/bc03/models_updated/Miles_Atlas/Chabrier_IMF/'
	models_dir_bc03 = '/Users/guidorb/AstroSoftware/bc03/models/Padova1994_bc03/chabrier/'
	models_result = data_path+'fits/'
	metal = ['m42', 'm52', 'm62', 'm72']
	year = ['0_005', '0_025', '0_1', '0_2', '0_6', '0_9', '1_4', '2_5', '5_0', '10_0']

	#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	#;;   Read in stacked spectrum and define continuum points over which   ;;;
	#;;   NOT to do the chi-squared fitting                                 ;;;
	#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
	os.chdir('/Users/guidorb/AstroSoftware/bc03/src_modified/')
	os.system('source $bc03/.bc_bash')
	
	catalog_new = {}
	for binning in catalog['bins']:
		stack = catalog[binning]['mean_stack']
		errors = catalog[binning]['stack_err']
		wavelength = catalog[binning]['stack_wav']
	
		indices = np.where((np.isnan(stack) == False) & (np.isnan(errors) == False))
		stack = stack[indices]
		errors = errors[indices]
		wav0 = wavelength[indices]	
	
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		#;;   Use downgrade resolution to broaden the spectra by the velocity dispersion, then read   ;;
		#;;   the resulting spectra in and rebin to the same wavelength array as the stacks...        ;;
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		vel_disp = catalog[binning]['median_properties']['median_veldisp']
		inc = catalog[binning]['median_properties']['median_inclination']
	
		print('[Using BC03 models...]')
		for met in metal:
			os.system('./downgrade_resolution '+models_dir_bc03+'bc2003_hr_stelib_'+met+'_chab_ssp.ised 3 '+str(vel_disp)+' '+models_result+'SSP_stelib_chab_'+met+'_stack'+name+'_'+samp+'_'+galtype)
			os.system('./galaxevpl '+models_result+'SSP_stelib_chab_'+met+'_stack'+name+'_'+samp+'_'+galtype+' 0.005,0.025,0.1,0.2,0.6,0.9,1.4,2.5,5.0,10.0 '+' '+str(min(wav0))+','+str(max(wav0))+',0.0,0.0,0.0 '+models_result+'SSP_stelib_chab_'+met+'_finalstack_'+name+'_'+samp+'_'+galtype+'.txt')

		templates_m42 = np.genfromtxt(models_result+'SSP_stelib_chab_'+metal[0]+'_finalstack_'+name+'_'+samp+'_'+galtype+'.txt')
		templates_m52 = np.genfromtxt(models_result+'SSP_stelib_chab_'+metal[1]+'_finalstack_'+name+'_'+samp+'_'+galtype+'.txt')
		templates_m62 = np.genfromtxt(models_result+'SSP_stelib_chab_'+metal[2]+'_finalstack_'+name+'_'+samp+'_'+galtype+'.txt')
		templates_m72 = np.genfromtxt(models_result+'SSP_stelib_chab_'+metal[3]+'_finalstack_'+name+'_'+samp+'_'+galtype+'.txt')
	
		templates = np.hstack((templates_m42, templates_m52[:,1:],templates_m62[:,1:],templates_m72[:,1:]))
	
	
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		#;;   Interpolate templates over stack wavelength array and create a mask for emission lines  ;;;
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		lines = [3728.82,3835.39,3868.71,3888.65,3967.41,4101.76,4340.47,4363.21,4861.33,4958.92,5006.84,5893.00,6300.30,6547.,6564.,6586.,6715.,6730.]
		linewidth = 20.0
	
		# Mask
		mask = np.ones_like(wav0)
		for l in lines:
			if l == 5893.0:
				# mask[((wav0<=l+7.0) & (wav0>=l-10.0))] = 0.0
				mask[((wav0>5875.67) & (wav0<5910.))] = 0.0
				continue
			mask[((wav0<=l+linewidth) & (wav0>=l-linewidth))] = 0.0
		# Interpolate
		temp_is = [interp1d(templates[:,0], t) for t in templates.T[1:]]
		new_temps = np.array([t(wav0) for t in temp_is])
	
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		#;;   Normalise templates   ;;;
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		idx_low = np.argmin(abs(wav0 - 5450.))
		idx_high = np.argmin(abs(wav0 - 5550.))
		t_norms = [np.median(t[idx_low:idx_high]) for t in new_temps]
		for ii, tn in enumerate(t_norms):
			new_temps[ii] = new_temps[ii] / tn
	
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		#;;   Set the uncertainties of the stack, where the emission lines are, to huuuuuge values   ;;;
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
		bigerrors = np.where(mask == 0.0)
		errors[bigerrors] = 1e15
	
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;
		#;;   Fit the spectrum   ;;;
		#;;;;;;;;;;;;;;;;;;;;;;;;;;;
		he_range = np.where((wav0 >= 5865.0) & (wav0 <= 5879.0))
	
		bounds = np.array([[0.,1.0] for i in range(40)])
		bounds = np.vstack((bounds, [[0.0, 3.0] for i in range(40)]))
		bounds = np.vstack((bounds, [0.0, max(stack[he_range])]))
		bounds = np.vstack((bounds, [0.0, 25.0]))
		bounds = np.vstack((bounds, [0.0, 1.1]))
		
		print('--------------------------------------------------------------') 
		print('Fitting continuum ('+str(binning)+'/'+str(np.size(catalog['bins']))+' bins)... [Battisti]')
		print('--------------------------------------------------------------') 
		fitted = False
		while (fitted != True):
			p0 = [np.random.rand() for i in range(40)]
			for i in range(40):
				p0.append(3.0*np.random.rand())
			p0.append(max(stack[he_range])*np.random.rand())
			p0.append(25.0*np.random.rand())
			p0.append(1.1*np.random.rand())
			try:
				popt, pcov = curve_fit(battisti, wav0[(errors > 0.)], stack[(errors > 0.)], p0=p0, bounds=bounds.T, sigma=errors[(errors > 0.)])
				fit = battisti_full(wav0, *popt)
				print("SUCCESS")
			except (RuntimeError, np.linalg.LinAlgError):
				print("FAILED... Trying again.")
				continue
			fitted = True

		fit = battisti_full(wav0, *popt)
		print('Done.')

		continuumfit = {'CB08_fit':fit, 'CB08_params':popt, 'CB08_stack':stack, 'CB08_stack_err':catalog[binning]['stack_err'][indices], 'CB08_wavelength':wav0}
		catalog_new[binning] = continuumfit
	
	# os.system('echo "Yo, your script is done! Get back to work, peasant." | mail -s "CONTINUUM JOB DONE" "guidorb@star.ucl.ac.uk"')
	if (save==True):
		print('Saving...')
		if (sn_limit!=None) & (adaptive==True):
			pickle.dump(catalog_new, open('/Users/guidorb/Dropbox/SDSS/stacked/SDSSstackcatalog_fitcont_'+name+'_'+dimension+'Dadaptive_SN'+str(sn_limit)+'_'+samp+'_'+galtype+'.p', 'wb'))
		else:
			pickle.dump(catalog_new, open('/Users/guidorb/Dropbox/SDSS/stacked/SDSSstackcatalog_fitcont_'+name+'_'+dimension+'D_'+samp+'_'+galtype+'.p', 'wb'))
		return catalog_new
	else:
		return catalog_new