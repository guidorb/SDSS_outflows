#######################################################
##	Program written to call catalogs of master SDSS  ##
##  catalogs or saved catalogs from stacks, 		 ##
##  continuum-fitting or NaD-fitting.				 ##
##	by G. W. Roberts-Borsani						 ##
##	17/03/2017										 ##
#######################################################

import pickle


def call_maincatalog(samp, galtype, option=None, highsfrd=False, sort=False, sortbin=None):
	# Add '_metal' to the end of the next line for the mass-metallicity relation stack
	path = '/Users/guidorb/Dropbox/MPA-JHU/MPA-JHU_catalog_DR7_'+samp+'_'+galtype+'.p'
	if (highsfrd is True) & (sort is False):
		sdss_cat_first = pickle.load(open(path, "rb"), encoding='latin1')
		cat = dict([key, sdss_cat_first[key][(sdss_cat_first['sn_ratio'] > 2.) & (sdss_cat_first['sfr_density'] > 0.1)]] for key in sdss_cat_first.keys())
	elif (highsfrd is False) & (sort is True):
		sdss_cat_first = pickle.load(open(path, "rb"), encoding='latin1')
		sdss_cat_initial = dict([key, sdss_cat_first[key][(sdss_cat_first['sn_ratio'] > 2.)]] for key in sdss_cat_first.keys())
		sort_index = np.argsort(sdss_cat_initial[sortbin])
		cat = dict([key, sdss_cat_initial[key][sort_index]] for key in sdss_cat_initial.keys())
	elif (highsfrd is True) & (sort is True):
		sdss_cat_first = pickle.load(open(path, "rb"), encoding='latin1')
		sdss_cat_initial = dict([key, sdss_cat_first[key][(sdss_cat_first['sn_ratio'] > 2.) & (sdss_cat_first['sfr_density'] > 0.1)]] for key in sdss_cat_first.keys())
		sort_index = np.argsort(sdss_cat_initial[sortbin])
		cat = dict([key, sdss_cat_initial[key][sort_index]] for key in sdss_cat_initial.keys())
	else:
		temp_cat = pickle.load(open(path, "rb"), encoding='latin1')
		cat = dict([key, temp_cat[key][(temp_cat['sn_ratio'] > 2.)]] for key in temp_cat.keys())
	return cat


def call_savedcatalog(name, samp, galtype, codepart, sn_limit=None, adaptive=None, dimension=None):
	if (sn_limit is not None) & (adaptive is True):
		path = '/Users/guidorb/Dropbox/SDSS/stacked/SDSSstackcatalog_'+codepart+'_'+name+'_'+dimension+'Dadaptive_SN'+str(sn_limit)+'_'+samp+'_'+galtype+'.p'
	else:
		path = '/Users/guidorb/Dropbox/SDSS/stacked/SDSSstackcatalog_'+codepart+'_'+name+'_'+dimension+'D_'+samp+'_'+galtype+'.p'
	cat = pickle.load(open(path, "rb"), encoding='latin1')
	return cat

