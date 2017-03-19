##################################################
##	Program written to stack SDSS DR7 spectra	##
##	by G. W. Roberts-Borsani					##
##	17/03/2017									##
##################################################

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import astropy.io.fits as fits
import extinction as ext
from random import sample
from . import catalogs as cts


nad_b = 5889.951
nad_r = 5895.924
hei = 	5875.67
c = 3.e5


def find_nearest(array, value):
	idx = (np.abs(array-value)).argmin()
	return array[idx]


def wavelength_array(lambda_min, lambda_max, dlam, z_low, z_high):
	wvstackl = float(int(lambda_min/(1.+z_low)))
	wvstackh = float(int(lambda_max/(1.+z_high)))
	rlamlimit = [wvstackl,wvstackh]
	dlam = 1.
	npoints = (rlamlimit[1]-rlamlimit[0])/dlam
	wavelength = np.arange(0.0, npoints, step=1.)*dlam+rlamlimit[0]
	return wavelength


def cdf_bin2d(xdata, ydata, xcdf_pc=None, ycdf_pc=None):
	# Cumulative distribution function (CDF) along x-axis
	xcdf = 1. * np.arange(len(xdata)) / (len(xdata) - 1)
	xsort = np.sort(xdata)

	# initialise arrays for x and y bin edges
	xbin = []
	ybin = [[] for y in range(len(xcdf_pc)-1)] 
	
	# find bin edges in x
	for i in range(len(xcdf_pc)):#
	    xbin.append(xsort[xcdf==find_nearest(xcdf,xcdf_pc[i])][0])

	# find bin edges in y
	for i in range(len(xbin)-1):
		idx = np.where((xdata>=xbin[i]) & (xdata<xbin[i+1]))
		ysort = np.sort(ydata[idx])
		ynew = ydata[idx]
		xnew = xdata[idx]
		# y data Cumulative Distribution Function (CDF)
		ycdf = 1. * np.arange(len(ysort)) / (len(ysort) - 1)
		for pc in ycdf_pc:
			y_pc = ysort[ycdf==find_nearest(ycdf,pc)]
			ybin[i].append(ysort[ycdf==find_nearest(ycdf,pc)][0])
	return xbin, ybin, xcdf


def cdf_bin2d_adaptive(xdata, ydata, cdf_pc=None, binax=None):
	# Cumulative distribution function (CDF) along x-axis
	if binax=='x':
		data = xdata.copy()
	if binax=='y':
		data = ydata.copy()
	cdf = 1. * np.arange(len(data)) / (len(data) - 1)
	sort = np.sort(data)

	# initialise arrays for x and y bin edges
	bins = []
	
	# find bin edges in x
	for i in range(len(cdf_pc)):#
		bins.append(sort[cdf==find_nearest(cdf,cdf_pc[i])][0])
	return bins, cdf


def get_bins(samp, galtype, xstring, ystring, plot=False, saveplot=False, optional_xbinsize=None, optional_ybinsize=None, xdatacoordinates=False, ydatacoordinates=False):
	cat = cts.call_maincatalog(samp, galtype)
	xdata = cat[xstring]
	ydata = cat[ystring]

	xcdf_pc = [0.,0.01,0.1,0.3,0.5,0.7,0.925,0.99,1.]
	ycdf_pc = [0,0.05,0.32,0.68,0.95,1]
	if (optional_xbinsize is not None) & (xdatacoordinates==False):
		xcdf_pc = optional_xbinsize.copy()
	if (optional_ybinsize is not None) & (ydatacoordinates==False):
		ycdf_pc = optional_ybinsize.copy()
	
	xbin,ybin,xcdf = cdf_bin2d(xdata, ydata, xcdf_pc, ycdf_pc)
	if (optional_xbinsize is not None) & (xdatacoordinates==True):
		xbin = optional_xbinsize
	if (optional_ybinsize is not None) & (ydatacoordinates==True):
		ybin = optional_ybinsize

	# If you want fixed values, define xbin or ybin as an array with the 
	# limits of the bins (for each bin of the opposite axis)..
	xbinfinal = np.array(xbin).copy()
	if (optional_ybinsize is not None) & (ydatacoordinates==True):
		ybinfinal = []
		for i in range(len(xbinfinal)-1):
			ybinfinal.append(ybin)
		ybinfinal = np.array(ybinfinal)
	else:
		ybinfinal = np.array(ybin).copy()

	if (plot is True):
		f, axarr = plt.subplots(3, sharex=True, figsize=(8,15))
		f.subplots_adjust(hspace=0)
		
		axarr[1].set_yticks(xcdf_pc[:-1])
		axarr[2].set_ylabel(ystring)
		axarr[2].set_xlabel(xstring)
		axarr[1].set_ylabel(r'CDF')
		axarr[0].set_ylabel(ystring)
		axarr[1].set_xlim(xdata.min(),xdata.max())
		# axarr[1].set_ylim(min(ydata),max(ydata))
		axarr[2].set_ylim(ydata.min()-1,ydata.max()+1)
		axarr[0].set_ylim(ydata.min()-1,ydata.max()+1)
		
		axarr[0].scatter(xdata,ydata,c='c',marker='o',alpha=0.2)
		
		# c = ['b','g','m','r','orange']
		for i in range(len(xcdf_pc)-1):
			idx = np.where((xdata>=xbin[i]) & (xdata<xbin[i+1]))
			ysort = np.sort(ydata[idx])
			ynew = ydata[idx]
			xnew = xdata[idx]
			axarr[2].scatter(xnew,ynew,color='dodgerblue',edgecolor='black',marker='o',alpha=0.2)
			# y data Cumulative Distribution Function (CDF)
			ycdf = 1. * np.arange(len(ysort)) / (len(ysort) - 1)
			for pc in xcdf_pc:
				y_pc = ysort[ycdf==find_nearest(ycdf,pc)]
				ybin[i].append(ysort[ycdf==find_nearest(ycdf,pc)][0])
		
		for i in range(len(xbin)-1):
			for j in range(len(ybin)-(len(xcdf_pc)-len(ycdf_pc))):
				axarr[2].vlines(xbin[i],ybin[i][j],ybin[i][j+1],linestyle='-')
				axarr[2].hlines(ybin[i][j],xbin[i],xbin[i+1],linestyle='-')
				axarr[2].vlines(xbin[i+1],ybin[i][j],ybin[i][j+1],linestyle='-')
				axarr[2].hlines(ybin[i][j+1],xbin[i],xbin[i+1],linestyle='-')
		
			
		axarr[1].plot(np.sort(xdata),xcdf, 'k-')
		axarr[1].vlines(xbin,[0],xcdf_pc,linestyle='--')
		axarr[1].hlines(xcdf_pc,[-4],xbin,linestyle='--')
		axarr[1].plot(xbin,xcdf_pc, 'or')
		if (saveplot==True):
			plt.savefig('/Users/guidorb/GoogleDrive/SDSS/figs/bins_'+xstring+'v'+ystring+'_'+galtype+'.pdf')
		plt.show()
	return xbinfinal, ybinfinal


def get_bins_2Dadaptive(samp, galtype, xstring, ystring, plot=False, saveplot=False, optional_binsize=None, datacoordinates=False, binax=None):	
	cat = cts.call_maincatalog(samp, galtype, highsfrd=False)
	xdata = cat[xstring]
	ydata = cat[ystring]

	# cdf_pc = [0,0.05,0.32,0.68,0.95,1]
	cdf_pc = [0,0.05,0.32,0.68,0.95,1]
	if (optional_binsize is not None) & (datacoordinates==False):
		cdf_pc = optional_binsize.copy()
	
	bins, cdf = cdf_bin2d_adaptive(xdata, ydata, cdf_pc=cdf_pc, binax=binax)
	binfinal = bins.copy()
	if (optional_binsize is not None) & (datacoordinates==True):
		binfinal = optional_binsize.copy()
		if binax=='y':
			if binfinal[0]<ydata.min():
				binfinal[0] = min(ydata).copy()
			if binfinal[-1]>ydata.max():
				binfinal[-1] = max(ydata).copy()
		if binax=='x':
			if binfinal[0]<xdata.min():
				binfinal[0] = min(xdata).copy()
			if binfinal[-1]>xdata.max():
				binfinal[-1] = max(xdata).copy()


	if (plot==True) & (datacoordinates==True):
		fig = plt.figure(figsize=(8.5, 7.))
		axarr = fig.add_subplot(111)		
		axarr.set_ylim(ydata.min()-1,ydata.max()+1)
		axarr.set_xlim(xdata.min()-1,xdata.max()+1)
		axarr.set_xlabel(xstring)
		axarr.set_ylabel(ystring)
		axarr.scatter(xdata, ydata, color='dodgerblue', edgecolor='black', marker='o', alpha=0.2)
		
		if binax=='y':
			for i in range(len(binfinal)-1):
				if (i <= (np.size(binfinal)-2)):
					axarr.hlines(binfinal[i], xdata.min()-1, xdata.max()+1, linestyle='-')
					axarr.hlines(binfinal[i+1], xdata.min()-1, xdata.max()+1, linestyle='-')
		
		if binax=='x':
			for i in range(len(binfinal)-1):
				if (i <= (np.size(binfinal)-2)):
					axarr.vlines(binfinal[i], ydata.min()-1, ydata.max()+1,linestyle='-')
					axarr.vlines(binfinal[i+1], ydata.min()-1, ydata.max()+1, linestyle='-')


	if (plot==True) & (datacoordinates==False):
		f, axarr = plt.subplots(3, sharex=True, figsize=(8,15))
		f.subplots_adjust(hspace=0)
		
		if binax=='y':
			axarr[1].set_xticks(cdf_pc[:-1])
		if binax=='x':
			axarr[1].set_yticks(cdf_pc[:-1])
		axarr[2].set_ylabel(ystring)
		axarr[2].set_xlabel(xstring)
		axarr[1].set_ylabel(r'CDF')
		axarr[0].set_ylabel(ystring)
		axarr[1].set_xlim(xdata.min(),xdata.max())
		axarr[2].set_ylim(ydata.min()-1,ydata.max()+1)
		axarr[0].set_ylim(ydata.min()-1,ydata.max()+1)
		
		axarr[0].scatter(xdata,ydata,c='c',marker='o',alpha=0.2)
		axarr[1].plot(binfinal,cdf_pc, 'or')
		
		# c = ['b','g','m','r','orange']
		if binax=='y':
			for i in range(len(binfinal)-1):
				idx = np.where((ydata>=binfinal[i]) & (ydata<binfinal[i+1]))
				xsort = np.sort(xdata[idx])
				ynew = ydata[idx]
				xnew = xdata[idx]
				axarr[2].scatter(xnew,ynew,color='dodgerblue',edgecolor='black',marker='o',alpha=0.2)
			
			for i in range(len(binfinal)-1):
				if (i <= (np.size(binfinal)-2)):
					axarr[2].hlines(binfinal[i], xdata.min()-1, xdata.max()+1,linestyle='-')
					axarr[2].hlines(binfinal[i+1], xdata.min()-1, xdata.max()+1, linestyle='-')

			axarr[1].plot(np.sort(ydata),cdf, 'k-')
			axarr[1].hlines(binfinal,[0],cdf_pc,linestyle='--')
		
		if binax=='x':
			for i in range(len(cdf_pc)-1):
				idx = np.where((xdata>=binfinal[i]) & (xdata<binfinal[i+1]))
				xsort = np.sort(ydata[idx])
				ynew = ydata[idx]
				xnew = xdata[idx]
				axarr[2].scatter(xnew,ynew,color='dodgerblue',edgecolor='black',marker='o',alpha=0.2)
			
			for i in range(len(binfinal)-1):
				if (i <= (np.size(binfinal)-2)):
					axarr[2].vlines(binfinal[i], ydata.min()-1, ydata.max()+1,linestyle='-')
					axarr[2].vlines(binfinal[i+1], ydata.min()-1, ydata.max()+1, linestyle='-')

			axarr[1].plot(np.sort(xdata),cdf, 'k-')
			axarr[1].vlines(binfinal,[0],cdf_pc,linestyle='--')
		
		if (saveplot==True):
			plt.savefig('/Users/guidorb/GoogleDrive/SDSS/figs/bins_'+xstring+'v'+ystring+'_'+galtype+'.pdf')
		plt.show()
	return binfinal


def plot_wbins(samp, galtype, xstring, ystring, xbins, ybins, secondgaltype=None):
	catalog = cts.call_maincatalog(samp, galtype, highsfrd=False)

	fig = plt.figure(figsize=(10., 8.))
	ax = fig.add_subplot(111)
	ax.scatter(catalog[xstring], catalog[ystring], color='deepskyblue', edgecolor='deepskyblue', marker='.')
	if (secondgaltype != None):
		catalog2 = cts.call_maincatalog(samp, secondgaltype, highsfrd=False)
		ax.scatter(catalog2[xstring], catalog2[ystring], color='purple', edgecolor='purple', marker='.')
	for i in range(len(xbins)-1):
		for j in range(len(ybins)-(len(xbins)-len(ybins[0]))):
			ax.vlines(xbins[i], ybins[i][j], ybins[i][j+1], linestyle='-', linewidth=1.75)
			ax.hlines(ybins[i][j], xbins[i], xbins[i+1], linestyle='-', linewidth=1.75)
			ax.vlines(xbins[i+1], ybins[i][j], ybins[i][j+1], linestyle='-', linewidth=1.75)
			ax.hlines(ybins[i][j+1], xbins[i], xbins[i+1], linestyle='-', linewidth=1.75)
	ax.set_xlim(min(xbins), max(xbins))
	ax.set_ylim(min(ybins.flatten()), max(ybins.flatten()))
	ax.set_xlabel(xstring, fontproperties=font)
	ax.set_ylabel(ystring, fontproperties=font)
	plt.show()
	return


def stacking_1d(name, samp, galtype, sn_limit, sort=False, sortbin=None, highsfrd=False, save=False, live=False, nstackstop=None):
	URL = 'http://das.sdss.org/spectro/1d_26/'
	path1 = '/Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'

	wavelength = wavelength_array(3800., 9200., 1., 0.05, 0.18)
	sdss_cat = cts.call_maincatalog(samp, galtype, highsfrd=highsfrd, sort=sort, sortbin=sortbin)

	fsum = np.zeros_like(wavelength)
	ferrsum = np.zeros_like(wavelength)
	totn = np.zeros_like(wavelength)
	temp_stack = []
	temp_err = []
	sumrange = []
	bins = []
	ibin = 0
	catalog = {}

	if live==True:
		plt.figure(figsize=(12.5, 10.))
		plt.axis([3500., 8000., 0.0, 2.5])
		plt.ion()
	for gal in range(len(sdss_cat[sortbin])):
		#################################################
		## Download spectrum if not already downloaded ##
		#################################################
		if (sdss_cat['plate'][gal] < 1000):
			if (sdss_cat['fiber'][gal] < 10):
				test = os.path.isfile(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==True:
					spec = fits.open(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==False:
					os.system('wget '+URL+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(sdss_cat['plate'][gal])+'/1d/')
					test = os.path.isfile(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit')
					if test==True:
						spec = fits.open(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit')
					else:
						continue
	
			if (sdss_cat['fiber'][gal] >= 10) & (sdss_cat['fiber'][gal] < 100):
				test = os.path.isfile(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==True:
					spec = fits.open(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==False:
					os.system('wget '+URL+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(sdss_cat['plate'][gal])+'/1d/')
					test = os.path.isfile(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit')
					if test==True:
						spec = fits.open(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit')
					else:
						continue
		
			if (sdss_cat['fiber'][gal] >= 100):
				test = os.path.isfile(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==True:
					spec = fits.open(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==False:
					os.system('wget '+URL+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(sdss_cat['plate'][gal])+'/1d/')
					test = os.path.isfile(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit')
					if test==True:
						spec = fits.open(path1+'0'+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-0'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit')
					else:
						continue
		
		if (sdss_cat['plate'][gal] >= 1000):
	
			if (sdss_cat['fiber'][gal] < 10):
				test = os.path.isfile(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==True:
					spec = fits.open(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==False:
					os.system('wget '+URL+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(sdss_cat['plate'][gal])+'/1d/')
					test = os.path.isfile(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit')
					if test==True:
						spec = fits.open(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-00'+str(sdss_cat['fiber'][gal])+'.fit')
					else:
						continue
		
			if (sdss_cat['fiber'][gal] >= 10) & (sdss_cat['fiber'][gal] < 100):
				test = os.path.isfile(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==True:
					spec = fits.open(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==False:
					os.system('wget '+URL+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(sdss_cat['plate'][gal])+'/1d/')
					test = os.path.isfile(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit')
					if test==True:
						spec = fits.open(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-0'+str(sdss_cat['fiber'][gal])+'.fit')
					else:
						continue
		
			if (sdss_cat['fiber'][gal] >= 100):
				test = os.path.isfile(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==True:
					spec = fits.open(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit')
				if test==False:
					os.system('wget '+URL+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(sdss_cat['plate'][gal])+'/1d/')
					test = os.path.isfile(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit')
					if test==True:
						spec = fits.open(path1+str(sdss_cat['plate'][gal])+'/1d/spSpec-'+str(sdss_cat['mjd'][gal])+'-'+str(sdss_cat['plate'][gal])+'-'+str(sdss_cat['fiber'][gal])+'.fit')
					else:
						continue
	
		sumrange.append(gal)
	
		###############################################################################
		## Get necessary arrays, correct for foreground dust using Schlegel dust and ##
		##  O'Donnell extinction law                                                 ##
		###############################################################################
		flux = spec[0].data[0]
		wav = 10**(spec[0].header['coeff0'] + spec[0].header['coeff1'] * np.arange(len(flux)))
		error = spec[0].data[2]
		hexaflags = spec[0].data[3]
		wav_air = wav/(1.0 + 2.735182e-4 + (131.4182/wav**2.0) + (2.76249e8/wav**4.0))
		
		Av = 3.1 * sdss_cat['E_BV_SFD'][gal]
		extinction = ext.odonnell(wav_air, Av)
		e_tau = extinction[1]
	
		restflux = flux * np.exp(e_tau) * (1+sdss_cat['z'][gal])
		restflerr = error * np.exp(e_tau) * (1+sdss_cat['z'][gal])
		restwl = wav_air/(1+sdss_cat['z'][gal])
	
	
		#######################
		## Normalise spectra ##
		#######################
		normalise = np.median(restflux[np.where((restwl >= 5450.0) & (restwl <= 5550.0))])
		fnorm = restflux/normalise
		ferrnorm = restflerr/normalise
	
	
		#######################
		## Define bad pixels ##
		#######################
		weights = np.zeros(len(restwl), dtype=int)
		weights[np.where(((hexaflags == 1.07374182e+09) | (hexaflags == 0.0)) & (ferrnorm > 0.0))] = 1
	
	
		#########################
		## Interpolate spectra ##
		#########################
		finterp = np.interp(wavelength, restwl, fnorm)
		ferrinterp = np.interp(wavelength, restwl, ferrnorm)
		winterp_temp = np.interp(wavelength, restwl, weights)
		winterp = []
		for intw in winterp_temp:
			winterp.append(int(round(intw)))
		winterp = np.array(winterp)
	
	
	
		####################################################################
		## Iteratively mean stack spectra to determine S/N ratio of stack ##
		####################################################################
		temp_stack.append(finterp * winterp)
		temp_err.append(ferrinterp * winterp)
	
		totn = totn + winterp
		fsum[(winterp == 1.)] = fsum[(winterp == 1.)] + (finterp[(winterp == 1.)] * winterp[(winterp == 1.)])
		ferrsum[(winterp == 1.)] = ferrsum[(winterp == 1.)] + (ferrinterp[(winterp == 1.)]**2.0)
	
		######################################
		## Compute signal-to-noise of stack ##
		######################################	
		cok = np.where( (((wavelength > 5820.) & (wavelength < 5850.)) | ((wavelength > 5920.) & (wavelength < 5950.))) & (fsum != 0.0) )
		sn_NaD = np.median(fsum[cok]/np.sqrt(ferrsum[cok]))
		print('Gal #: '+str(gal+1)+'/'+str(len(sdss_cat[sortbin]))+'             Measured SNR: '+str(sn_NaD)+'               '+sortbin+': '+str(sdss_cat[sortbin][gal]))
	
		if live==True:
			plt.plot(wavelength[(totn > 0)], fsum[(totn > 0)]/totn[(totn > 0)], color='blue', linewidth=1.5)
			plt.xlim(3500., 8000.)
			plt.ylim(0.0, 2.5)
			plt.text(6000., 0.3, s='S/N ratio = '+str(sn_NaD), fontsize=12)
			plt.pause(0.01)
			if ((gal+1)==len(sdss_cat[sortbin])):
				continue
			else:
				plt.cla()

	
		#####################################################################
		## Tell code when to stop stacking and move over to the next stack ##
		#####################################################################
		if (sn_NaD >= sn_limit):
						
			ibin = ibin + 1
			totn = np.array(totn)
			nozero = np.where(totn > 0)
			#################################################################################
			## Mean and median stack the spectra, and bootstrap errors to get stack errors ##
			#################################################################################
			fsum_mean = fsum[nozero]/totn[nozero]
			temp_wav = wavelength[nozero].copy()
			
			stack_array = np.array(temp_stack).copy().T
			error_array = np.array(temp_err).copy().T
			ferr = []
			fsum_median = []

			for fl, er, tot in zip(stack_array, error_array, totn):
				if (tot==0):
					continue
				else:
					new_fl = fl[(fl > 0.0)].copy()
					new_err = er[(fl > 0.0)].copy()
					if (np.size(new_fl) == 1):
						ferr.append(new_err)
					else:
						fsum_median.append(np.median(new_fl))
						med = []
						for xtimes in range(50):
							med.append(np.median(np.random.choice(new_fl, size=np.size(new_fl), replace=True)))
						ferr.append(np.std(med))
			ferr = np.array(ferr)
			fsum_median = np.array(fsum_median)

			if np.size(np.where(ferr == 0.)) > 0:
				print('Zeros in error array...')
	
	
			###############################
			## Get properties of the bin ##
			###############################
			sumrange = np.array(sumrange)
			properties = {}
			for key in sdss_cat.keys():
				if (key != 'ra') & (key != 'dec') & (key != 'sdsstype') & (key != 'objid') & (key != 'specobjid') & (key != 'plate') & (key != 'mjd') & (key != 'fiber') & (key != 'specclass') & (key != 'zwarning') & (key != 'schlegel_type'):
					medianname = 'median_'+key
					properties[medianname] = np.median(sdss_cat[key][sumrange])
			median_halpha = np.median(np.array(sdss_cat['halpha'][sumrange]))

			R_v = 4.05
			Qfit_ha = np.polyval([0.0145, -0.261, 1.803, -2.488], [1./(6562.8/5500.)])
			Qfit_hb = np.polyval([0.0145, -0.261, 1.803, -2.488], [1./(4861.33/5500.)])
			klam_ha = 2.396 * Qfit_ha + R_v
			klam_hb = 2.396 * Qfit_hb + R_v
			tauv = np.log((properties['median_halpha']/properties['median_hbeta'])/2.86)
			e_bv = (1.086 * tauv) / (klam_hb - klam_ha)
			av = e_bv * R_v
			
			median_hahb_extinct = av
			properties['median_balmer_dust'] = av
			
			
			###############################
			## Print and save everything ##
			###############################
			print('Number of gals in stack: '+str(np.size(sumrange)))
			print('Median velocity dispersion: '+str(properties['median_veldisp'])+' km/s')
			print('Median SFR density: '+str(properties['median_sfr_density_log'])+' Msol/yr/kpc^2')
			print('---------------------------------------------------------------')
			
			stackbin = {'mean_stack':fsum_mean, 'median_stack':fsum_median, 'stack_err':ferr, 'stack_wav':temp_wav,
						'ngals':np.size(sumrange), 'S/N_ratio':sn_NaD, 'totn':totn, 'median_properties':properties
						}
	
			binname = 'bin'+str(ibin)
			bins.append(binname)
			catalog[binname] = stackbin
	
			######################
			## Reset bin arrays ##
			######################
			totn = np.zeros_like(wavelength)
			fsum = np.zeros_like(wavelength)
			ferrsum = np.zeros_like(wavelength)
			temp_stack = []
			temp_err = []
			sumrange = []
			
			if (nstackstop == ibin):
				catalog['bins'] = np.array(bins)
				if save==True:
					pickle.dump(catalog, open('/Users/guidorb/GoogleDrive/SDSS/stacked/SDSSstackcatalog_stack_'+name+'_1D_'+samp+'_'+galtype+'.p', 'wb'))
				break
				return catalog

	catalog['bins'] = np.array(bins)
	os.system('echo "Yo, your script is done! Get back to work, peasant." | mail -s "STACKING JOB DONE" "guidorb@star.ucl.ac.uk"')
	if save==True:
		pickle.dump(catalog, open('/Users/guidorb/GoogleDrive/SDSS/stacked/SDSSstackcatalog_stack_'+name+'_1D_'+samp+'_'+galtype+'.p', 'wb'))
	return catalog


def stacking_2d(name, samp, galtype, xbinned, ybinned, sn_limit, highsfrd=False, save=False, live=False, nstackstop=None, plot=False, saveplot=False, xbins=None, ybins=None):
	URL = 'http://das.sdss.org/spectro/1d_26/'
	path1 = '/Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'

	wavelength = wavelength_array(3800., 9200., 1., 0.05, 0.18)
	sdss_cat = cts.call_maincatalog(samp, galtype, highsfrd=highsfrd)

	bins = []
	ibin = 0
	catalog = {}

	##################################################################
	## Get bin limits for 2D stacking and check if there are enough ##
	## galaxies in each bin..                                       ##
	##################################################################
	if (xbins is None) & (ybins is None):
		xbin, ybin = get_bins(samp, galtype, xbinned, ybinned, plot=plot, saveplot=saveplot)
	else:
		xbin=xbins
		ybin=ybins

	for i in range(len(xbin)):
		if (i <= (np.size(xbin)-2)):
			for j in range(len(ybin[i])):
				if (j <= (np.size(ybin[i])-2)):
					selection = np.where(	
										(sdss_cat[xbinned] >= xbin[i]) & (sdss_cat[xbinned] < xbin[i+1]) &
										(sdss_cat[ybinned] >= ybin[i][j]) & (sdss_cat[ybinned] < ybin[i][j+1])
										)
					ngals = np.size(selection)
					ibin = ibin + 1

					print('bin'+str(ibin)+': '+str(ngals)+' galaxies')

	var = input('Binning ok? [yes/no]       ')
	print('                   ')
	if var=='no':
		return 'Binning not good: change and try again...'
	if var=='yes':
		ibin = 0
		for i in range(len(xbin)):
			if (i <= (np.size(xbin)-2)):
				for j in range(len(ybin[i])):
					if (j <= (np.size(ybin[i])-2)):
						selection = np.where(	
											(sdss_cat[xbinned] >= xbin[i]) & (sdss_cat[xbinned] < xbin[i+1]) &
											(sdss_cat[ybinned] >= ybin[i][j]) & (sdss_cat[ybinned] < ybin[i][j+1])
											)
						ngals = np.size(selection)
						if (ngals<=3):
							continue

						sdss_sample = dict([key, sdss_cat[key][selection]] for key in sdss_cat.keys())
	
						fsum = np.zeros_like(wavelength)
						ferrsum = np.zeros_like(wavelength)
						totn = np.zeros_like(wavelength)
						temp_stack = []
						temp_err = []
						sumrange = []

						#################################################
						## Get necessary properties from random sample ##
						#################################################
						x = sample(range(0,ngals), ngals)

						z = np.array(sdss_sample['z'][x])
						vdisp = np.array(sdss_sample['veldisp'][x])
						inc = np.array(sdss_sample['inclination'][x])
						sn_ratio_spec = np.array(sdss_sample['sn_ratio'][x])
						Ebv = np.array(sdss_sample['E_BV_SFD'][x])
						plate = np.array(sdss_sample['plate'][x])
						mjd = np.array(sdss_sample['mjd'][x])
						fiber = np.array(sdss_sample['fiber'][x])
						sfrd_log = np.array(sdss_sample['sfr_density_log'][x])
						mass = np.array(sdss_sample['totmass'][x])
						d4000 = np.array(sdss_sample['d4000'][x])
						halpha = np.array(sdss_sample['halpha'][x])
						hbeta = np.array(sdss_sample['hbeta'][x])
						oiii = np.array(sdss_sample['oiii'][x])
						nii = np.array(sdss_sample['nii'][x])
						dust_internal = np.array(sdss_sample['dust_av'][x])
						totsfr = np.array(sdss_sample['totsfr'][x])

						for gal in range(len(x)):
							if (
								((plate[gal]==635) & (mjd[gal]==52145) & (fiber[gal]==287)) | 
								((plate[gal]==768) & (mjd[gal]==52281) & (fiber[gal]==159))
								) & (galtype == 'SF') & (samp == 'fullSDSS'):
								continue
							#################################################
							## Download spectrum if not already downloaded ##
							#################################################
							if (plate[gal] < 1000):
								if (fiber[gal] < 10):
									test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
									if test==True:
										spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
									if test==False:
										os.system('wget '+URL+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(plate[gal])+'/1d/')
										test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
										if test==True:
											spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
										else:
											continue
				
								if (fiber[gal] >= 10) & (fiber[gal] < 100):
									test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
									if test==True:
										spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
									if test==False:
										os.system('wget '+URL+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(plate[gal])+'/1d/')
										test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
										if test==True:
											spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
										else:
											continue
							
								if (fiber[gal] >= 100):
									test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
									if test==True:
										spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
									if test==False:
										os.system('wget '+URL+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(plate[gal])+'/1d/')
										test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
										if test==True:
											spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
										else:
											continue
							
							if (plate[gal] >= 1000):
				
								if (fiber[gal] < 10):
									test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
									if test==True:
										spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
									if test==False:
										os.system('wget '+URL+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(plate[gal])+'/1d/')
										test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
										if test==True:
											spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
										else:
											continue
							
								if (fiber[gal] >= 10) & (fiber[gal] < 100):
									test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
									if test==True:
										spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
									if test==False:
										os.system('wget '+URL+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(plate[gal])+'/1d/')
										test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
										if test==True:
											spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
										else:
											continue
							
								if (fiber[gal] >= 100):
									test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
									if test==True:
										spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
									if test==False:
										os.system('wget '+URL+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(plate[gal])+'/1d/')
										test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
										if test==True:
											spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
										else:
											continue
				
							sumrange.append(x[gal])
				
							###############################################################################
							## Get necessary arrays, correct for foreground dust using Schlegel dust and ##
							##  O'Donnell extinction law                                                 ##
							###############################################################################
							flux = spec[0].data[0]
							wav = 10**(spec[0].header['coeff0'] + spec[0].header['coeff1'] * np.arange(len(flux)))
							error = spec[0].data[2]
							hexaflags = spec[0].data[3]
							wav_air = wav/(1.0 + 2.735182e-4 + (131.4182/wav**2.0) + (2.76249e8/wav**4.0))
							
							Av = 3.1 * Ebv[gal]
							extinction = ext.odonnell(wav_air, Av)
							e_tau = extinction[1]
						
							restflux = flux * np.exp(e_tau) * (1+z[gal])
							restflerr = error * np.exp(e_tau) * (1+z[gal])
							restwl = wav_air/(1+z[gal])
						
						
							#######################
							## Normalise spectra ##
							#######################
							normalise = np.median(restflux[np.where((restwl >= 5450.0) & (restwl <= 5550.0))])
							fnorm = restflux/normalise
							ferrnorm = restflerr/normalise
						
						
							#######################
							## Define bad pixels ##
							#######################
							weights = np.zeros(len(restwl), dtype=int)
							weights[np.where(((hexaflags == 1.07374182e+09) | (hexaflags == 0.0)) & (ferrnorm > 0.0))] = 1
						
						
							#########################
							## Interpolate spectra ##
							#########################
							finterp = np.interp(wavelength, restwl, fnorm)
							ferrinterp = np.interp(wavelength, restwl, ferrnorm)
							winterp_temp = np.interp(wavelength, restwl, weights)
							winterp = []
							for intw in winterp_temp:
								winterp.append(int(round(intw)))
							winterp = np.array(winterp)
						
						
							####################################################################
							## Iteratively mean stack spectra to determine S/N ratio of stack ##
							####################################################################
							temp_stack.append(finterp * winterp)
							temp_err.append(ferrinterp * winterp)
						
							totn[(winterp == 1.)] = totn[(winterp == 1.)] + winterp[(winterp == 1.)]
							fsum[(winterp == 1.)] = fsum[(winterp == 1.)] + (finterp[(winterp == 1.)] * winterp[(winterp == 1.)])
							ferrsum[(winterp == 1.)] = ferrsum[(winterp == 1.)] + (ferrinterp[(winterp == 1.)]**2.0)
						

							######################################
							## Compute signal-to-noise of stack ##
							######################################	
							cok = np.where( (((wavelength > 5820.) & (wavelength < 5850.)) | ((wavelength > 5920.) & (wavelength < 5950.))) & (fsum > 0.0) )
							sn_NaD = np.median(fsum[cok]/np.sqrt(ferrsum[cok]))
							print('Gal #: '+str(gal+1)+'/'+str(len(x))+'             Measured SNR: '+str(sn_NaD))
				
							if live==True:
								plt.plot(wavelength[(totn > 0)], fsum[(totn > 0)]/totn[(totn > 0)], color='blue', linewidth=1.5)
								plt.xlim(3500., 8000.)
								plt.ylim(0.0, 2.5)
								plt.text(6000., 0.3, s='S/N ratio = '+str(sn_NaD), fontsize=12)
								plt.pause(0.01)
								if ((gal+1)==len(x)):
									continue
								else:
									plt.cla()
				

							#####################################################################
							## Tell code when to stop stacking and move over to the next stack ##
							#####################################################################
							if (gal == (len(x)-1)):
						
								ibin = ibin + 1
								totn = np.array(totn)
								nozero = np.where(totn > 0)
								#################################################################################
								## Mean and median stack the spectra, and bootstrap errors to get stack errors ##
								#################################################################################
								fsum_mean = fsum[nozero]/totn[nozero]
								temp_wav = wavelength[nozero].copy()
						
								stack_array = np.array(temp_stack).copy().T
								error_array = np.array(temp_err).copy().T
								ferr = []
								fsum_median = []

								for fl, er, tot in zip(stack_array, error_array, totn):
									if (tot==0):
										continue
									else:
										new_fl = fl[(fl > 0.0)].copy()
										new_err = er[(fl > 0.0)].copy()
										if (np.size(new_fl) == 1):
											ferr.append(new_err)
										else:
											fsum_median.append(np.median(new_fl))
											med = []
											for xtimes in range(50):
												med.append(np.median(np.random.choice(new_fl, size=np.size(new_fl), replace=True)))
											ferr.append(np.std(med))
								ferr = np.array(ferr)
								fsum_median = np.array(fsum_median)

								if np.size(np.where(ferr == 0.)) > 0:
									print('Zeros in error array...')

						
								###############################
								## Get properties of the bin ##
								###############################
								sumrange = np.array(sumrange)
								properties = {}
								for key in sdss_sample.keys():
									if (key != 'ra') & (key != 'dec') & (key != 'sdsstype') & (key != 'objid') & (key != 'specobjid') & (key != 'plate') & (key != 'mjd') & (key != 'fiber') & (key != 'specclass') & (key != 'zwarning') & (key != 'schlegel_type'):
										medianname = 'median_'+key
										properties[medianname] = np.median(sdss_sample[key][sumrange])
								median_halpha = np.median(np.array(sdss_sample['halpha'][sumrange]))

								R_v = 4.05
								Qfit_ha = np.polyval([0.0145, -0.261, 1.803, -2.488], [1./(6562.8/5500.)])
								Qfit_hb = np.polyval([0.0145, -0.261, 1.803, -2.488], [1./(4861.33/5500.)])
								klam_ha = 2.396 * Qfit_ha + R_v
								klam_hb = 2.396 * Qfit_hb + R_v
								tauv = np.log((properties['median_halpha']/properties['median_hbeta'])/2.86)
								e_bv = (1.086 * tauv) / (klam_hb - klam_ha)
								av = e_bv * R_v
						
								median_hahb_extinct = av
								properties['median_balmer_dust'] = av
				
						
								###############################
								## Print and save everything ##
								###############################
								print('Number of gals in stack: '+str(np.size(sumrange)))
								print('Median velocity dispersion: '+str(properties['median_veldisp'])+' km/s')
								print('Median SFR density: '+str(properties['median_sfr_density_log'])+' Msol/yr/kpc^2')
								print('---------------------------------------------------------------')
						
								stackbin = {'mean_stack':fsum_mean, 'median_stack':fsum_median, 'stack_err':ferr, 'stack_wav':temp_wav,
											'ngals':np.size(sumrange), 'S/N_ratio':sn_NaD, 'totn':totn, 'median_properties':properties
											}
						
								binname = 'bin'+str(ibin)
								bins.append(binname)
								catalog[binname] = stackbin
						
								######################
								## Reset bin arrays ##
								######################
								totn = np.zeros_like(wavelength)
								fsum = np.zeros_like(wavelength)
								ferrsum = np.zeros_like(wavelength)
								temp_stack = []
								temp_err = []
								sumrange = []

								continue

						if (nstackstop == ibin):
							catalog['bins'] = np.array(bins)
							catalog['xbins'] = xbins
							catalog['ybins'] = ybins
							if save==True:
								pickle.dump(catalog, open('/Users/guidorb/GoogleDrive/SDSS/stacked/SDSSstackcatalog_stack_'+name+'_2D_'+samp+'_'+galtype+'.p', 'wb'))
							break
							return catalog
				
	catalog['bins'] = np.array(bins)
	catalog['xbins'] = xbins
	catalog['ybins'] = ybins
	# os.system('echo "Yo, your script is done! Get back to work, peasant." | mail -s "STACKING JOB DONE" "guidorb@star.ucl.ac.uk"')
	if save==True:
		print('Saving..')
		pickle.dump(catalog, open('/Users/guidorb/GoogleDrive/SDSS/stacked/SDSSstackcatalog_stack_'+name+'_2D_'+samp+'_'+galtype+'.p', 'wb'))
	return catalog


def stacking_2d_adaptive(name, samp, galtype, xbinned, ybinned, sn_limit, sortbin=False, highsfrd=False, save=False, live=False, nstackstop=None, plot=False, saveplot=False, b=None, binax=None):
	URL = 'http://das.sdss.org/spectro/1d_26/'
	path1 = '/Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'

	wavelength = wavelength_array(3800., 9200., 1., 0.05, 0.18)
	sdss_cat = cts.call_maincatalog(samp, galtype, highsfrd=highsfrd)

	###########################################################################
	## Get bin limits for 2D adaptive stacking and check if there are enough ##
	## galaxies in each bin..                                                ##
	###########################################################################
	if (b is None):
		binss = get_bins_2Dadaptive(samp, galtype, xbinned, ybinned, plot=plot, saveplot=saveplot, binax=binax)
		b = binss.copy()


	ibin = 0
	for j in range(len(b)):
		if (j <= (np.size(b)-2)):
			if binax=='x':
				selection = np.where(	
									(sdss_cat[xbinned] >= b[j]) & (sdss_cat[xbinned] < b[j+1])
									)
			if binax=='y':
				selection = np.where(	
									(sdss_cat[ybinned] >= b[j]) & (sdss_cat[ybinned] < b[j+1])
									)
			ngals = np.size(selection)
			ibin = ibin + 1

			print('bin'+str(ibin)+': '+str(ngals)+' galaxies')

	var = input('Binning ok? [yes/no]       ')
	print('                   ')
	if var=='no':
		return 'Binning not good: change and try again...'
	if var=='yes':
		ibin = 0
		bins = []
		catalog = {}

		########################
		##   Start stacking   ##
		########################
		for j in range(len(b)):
			if (j <= (np.size(b)-2)):
				# Get the sample defined by the y-axis bin (eg., SFR)
				if binax=='x':
					selection = np.where(	
										(sdss_cat[xbinned] >= b[j]) & (sdss_cat[xbinned] < b[j+1])
										)
				if binax=='y':
					selection = np.where(	
										(sdss_cat[ybinned] >= b[j]) & (sdss_cat[ybinned] < b[j+1])
										)
				ngals = np.size(selection)
				if (ngals<=3):
					continue

				# Sort the obtained sample in order of the non-binned quantity (eg., SFRs)
				sdss_sample_initial = dict([key, sdss_cat[key][selection]] for key in sdss_cat.keys())
				if (sortbin is not False):
					sort_index = np.argsort(sdss_sample_initial[sortbin])
					sdss_sample = dict([key, sdss_sample_initial[key][sort_index]] for key in sdss_sample_initial.keys())
	
				# Prepare arrays necessary for stacking
				totn = np.zeros_like(wavelength)
				fsum = np.zeros_like(wavelength)
				ferrsum = np.zeros_like(wavelength)
				temp_stack = []
				temp_err = []
				sumrange = []

				#################################################
				## Get necessary properties from random sample ##
				#################################################

				z = np.array(sdss_sample['z'])
				vdisp = np.array(sdss_sample['veldisp'])
				inc = np.array(sdss_sample['inclination'])
				sn_ratio_spec = np.array(sdss_sample['sn_ratio'])
				Ebv = np.array(sdss_sample['E_BV_SFD'])
				plate = np.array(sdss_sample['plate'])
				mjd = np.array(sdss_sample['mjd'])
				fiber = np.array(sdss_sample['fiber'])
				sfrd_log = np.array(sdss_sample['sfr_density_log'])
				mass = np.array(sdss_sample['totmass'])
				d4000 = np.array(sdss_sample['d4000'])
				halpha = np.array(sdss_sample['halpha'])
				hbeta = np.array(sdss_sample['hbeta'])
				oiii = np.array(sdss_sample['oiii'])
				nii = np.array(sdss_sample['nii'])
				dust_internal = np.array(sdss_sample['dust_av'])
				totsfr = np.array(sdss_sample['totsfr'])

				if live==True:
					plt.figure(figsize=(12.5, 10.))
					plt.axis([3500., 8000., 0.0, 2.5])
					plt.ion()
				sumn = 0
				for gal in range(len(z)):
					# Small except for a couple of spectra which seem corrupted and can't be stacked
					if (
						((plate[gal]==635) & (mjd[gal]==52145) & (fiber[gal]==287)) | 
						((plate[gal]==768) & (mjd[gal]==52281) & (fiber[gal]==159))
						) & (galtype == 'SF') & (samp == 'fullSDSS'):
						continue
					#################################################
					## Download spectrum if not already downloaded ##
					#################################################
					if (plate[gal] < 1000):
						if (fiber[gal] < 10):
							test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
							if test==True:
								spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
							if test==False:
								os.system('wget '+URL+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(plate[gal])+'/1d/')
								test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
								if test==True:
									spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
								else:
									continue
		
						if (fiber[gal] >= 10) & (fiber[gal] < 100):
							test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
							if test==True:
								spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
							if test==False:
								os.system('wget '+URL+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(plate[gal])+'/1d/')
								test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
								if test==True:
									spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
								else:
									continue
					
						if (fiber[gal] >= 100):
							test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
							if test==True:
								spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
							if test==False:
								os.system('wget '+URL+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/0'+str(plate[gal])+'/1d/')
								test = os.path.isfile(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
								if test==True:
									spec = fits.open(path1+'0'+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-0'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
								else:
									continue
					
					if (plate[gal] >= 1000):
		
						if (fiber[gal] < 10):
							test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
							if test==True:
								spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
							if test==False:
								os.system('wget '+URL+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(plate[gal])+'/1d/')
								test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
								if test==True:
									spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-00'+str(fiber[gal])+'.fit')
								else:
									continue
					
						if (fiber[gal] >= 10) & (fiber[gal] < 100):
							test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
							if test==True:
								spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
							if test==False:
								os.system('wget '+URL+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(plate[gal])+'/1d/')
								test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
								if test==True:
									spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-0'+str(fiber[gal])+'.fit')
								else:
									continue
					
						if (fiber[gal] >= 100):
							test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
							if test==True:
								spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
							if test==False:
								os.system('wget '+URL+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit -P /Users/guidorb/PhD/Outflows/SDSS/spectro/1d_26/'+str(plate[gal])+'/1d/')
								test = os.path.isfile(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
								if test==True:
									spec = fits.open(path1+str(plate[gal])+'/1d/spSpec-'+str(mjd[gal])+'-'+str(plate[gal])+'-'+str(fiber[gal])+'.fit')
								else:
									continue
					
					sumrange.append(sumn)
		
					###############################################################################
					## Get necessary arrays, correct for foreground dust using Schlegel dust and ##
					##  O'Donnell extinction law                                                 ##
					###############################################################################
					flux = spec[0].data[0]
					wav = 10**(spec[0].header['coeff0'] + spec[0].header['coeff1'] * np.arange(len(flux)))
					error = spec[0].data[2]
					hexaflags = spec[0].data[3]
					wav_air = wav/(1.0 + 2.735182e-4 + (131.4182/wav**2.0) + (2.76249e8/wav**4.0))
					
					Av = 3.1 * Ebv[gal]
					extinction = ext.odonnell(wav_air, Av)
					e_tau = extinction[1]
				
					restflux = flux * np.exp(e_tau) * (1+z[gal])
					restflerr = error * np.exp(e_tau) * (1+z[gal])
					restwl = wav_air/(1+z[gal])
				
				
					#######################
					## Normalise spectra ##
					#######################
					normalise = np.median(restflux[np.where((restwl >= 5450.0) & (restwl <= 5550.0))])
					fnorm = restflux/normalise
					ferrnorm = restflerr/normalise
				
				
					#######################
					## Define bad pixels ##
					#######################
					weights = np.zeros(len(restwl), dtype=int)
					weights[np.where(((hexaflags == 1.07374182e+09) | (hexaflags == 0.0)) & (ferrnorm > 0.0))] = 1
				
				
					#########################
					## Interpolate spectra ##
					#########################
					finterp = np.interp(wavelength, restwl, fnorm)
					ferrinterp = np.interp(wavelength, restwl, ferrnorm)
					winterp_temp = np.interp(wavelength, restwl, weights)
					winterp = []
					for intw in winterp_temp:
						winterp.append(int(round(intw)))
					winterp = np.array(winterp)
				
				
					####################################################################
					## Iteratively mean stack spectra to determine S/N ratio of stack ##
					####################################################################
					temp_stack.append(finterp * winterp)
					temp_err.append(ferrinterp * winterp)
				
					totn[(winterp == 1.)] = totn[(winterp == 1.)] + winterp[(winterp == 1.)]
					fsum[(winterp == 1.)] = fsum[(winterp == 1.)] + (finterp[(winterp == 1.)] * winterp[(winterp == 1.)])
					ferrsum[(winterp == 1.)] = ferrsum[(winterp == 1.)] + (ferrinterp[(winterp == 1.)]**2.0)
				

					######################################
					## Compute signal-to-noise of stack ##
					######################################	
					cok = np.where( (((wavelength > 5820.) & (wavelength < 5850.)) | ((wavelength > 5920.) & (wavelength < 5950.))) & (fsum > 0.0) )
					sn_NaD = np.median(fsum[cok]/np.sqrt(ferrsum[cok]))
					print('Gal #: '+str(gal+1)+'/'+str(len(z))+'             Measured SNR: '+str(sn_NaD))
		
					if live==True:
						plt.plot(wavelength[(totn > 0)], fsum[(totn > 0)]/totn[(totn > 0)], color='blue', linewidth=1.5)
						plt.xlim(3500., 8000.)
						plt.ylim(0.0, 2.5)
						plt.text(6000., 0.3, s='S/N ratio = '+str(sn_NaD), fontsize=12)
						plt.pause(0.01)
						plt.cla()
		

					#####################################################################
					## Tell code when to stop stacking and move over to the next stack ##
					#####################################################################
					sumn = sumn + 1
					if (sn_NaD >= sn_limit):
				
						ibin = ibin + 1
						totn = np.array(totn)
						nozero = np.where(totn > 0)
						#################################################################################
						## Mean and median stack the spectra, and bootstrap errors to get stack errors ##
						#################################################################################
						fsum_mean = fsum[nozero]/totn[nozero]
						temp_wav = wavelength[nozero].copy()
				
						stack_array = np.array(temp_stack).copy().T
						error_array = np.array(temp_err).copy().T
						ferr = []
						fsum_median = []

						for fl, er, tot in zip(stack_array, error_array, totn):
							if (tot==0):
								continue
							else:
								new_fl = fl[(fl > 0.0)].copy()
								new_err = er[(fl > 0.0)].copy()
								if (np.size(new_fl) == 1):
									ferr.append(new_err)
								else:
									fsum_median.append(np.median(new_fl))
									med = []
									for xtimes in range(50):
										med.append(np.median(np.random.choice(new_fl, size=np.size(new_fl), replace=True)))
									ferr.append(np.std(med))
						ferr = np.array(ferr)
						fsum_median = np.array(fsum_median)

						if np.size(np.where(ferr == 0.)) > 0:
							print('Zeros in error array...')

				
						###############################
						## Get properties of the bin ##
						###############################
						sumrange = np.array(sumrange)
						properties = {}
						for key in sdss_sample.keys():
							if (key != 'ra') & (key != 'dec') & (key != 'sdsstype') & (key != 'objid') & (key != 'specobjid') & (key != 'plate') & (key != 'mjd') & (key != 'fiber') & (key != 'specclass') & (key != 'zwarning') & (key != 'schlegel_type'):
								medianname = 'median_'+key
								properties[medianname] = np.median(sdss_sample[key][sumrange])
						median_halpha = np.median(np.array(sdss_sample['halpha'][sumrange]))

						R_v = 4.05
						Qfit_ha = np.polyval([0.0145, -0.261, 1.803, -2.488], [1./(6562.8/5500.)])
						Qfit_hb = np.polyval([0.0145, -0.261, 1.803, -2.488], [1./(4861.33/5500.)])
						klam_ha = 2.396 * Qfit_ha + R_v
						klam_hb = 2.396 * Qfit_hb + R_v
						tauv = np.log((properties['median_halpha']/properties['median_hbeta'])/2.86)
						e_bv = (1.086 * tauv) / (klam_hb - klam_ha)
						av = e_bv * R_v
				
						median_hahb_extinct = av
						properties['median_balmer_dust'] = av
		
				
						###############################
						## Print and save everything ##
						###############################
						print('Number of gals in stack: '+str(np.size(sumrange)))
						print('Median velocity dispersion: '+str(properties['median_veldisp'])+' km/s')
						print('Median SFR density: '+str(properties['median_sfr_density_log'])+' Msol/yr/kpc^2')
						print('---------------------------------------------------------------')
				
						stackbin = {'mean_stack':fsum_mean, 'median_stack':fsum_median, 'stack_err':ferr, 'stack_wav':temp_wav,
									'ngals':np.size(sumrange), 'S/N_ratio':sn_NaD, 'totn':totn, 'median_properties':properties
									}
				
						binname = 'bin'+str(ibin)
						bins.append(binname)
						catalog[binname] = stackbin
				
						######################
						## Reset bin arrays ##
						######################
						totn = np.zeros_like(wavelength)
						fsum = np.zeros_like(wavelength)
						ferrsum = np.zeros_like(wavelength)
						temp_stack = []
						temp_err = []
						sumrange = []
						sumn = 0

						# continue

						if (nstackstop == ibin):
							catalog['bins'] = np.array(bins)
							catalog['xbins'] = b
							catalog['ybins'] = b
							if save==True:
								pickle.dump(catalog, open('/Users/guidorb/GoogleDrive/SDSS/stacked/SDSSstackcatalog_stack_'+name+'_2Dadaptive_SN'+str(sn_limit)+'_'+samp+'_'+galtype+'.p', 'wb'))
							break
							return catalog
		
		catalog['bins'] = np.array(bins)
		catalog['xbins'] = b
		catalog['ybins'] = b
		# os.system('echo "Yo, your script is done! Get back to work, peasant." | mail -s "STACKING JOB DONE" "guidorb@star.ucl.ac.uk"')
		if save==True:
			print('Saving..')
			pickle.dump(catalog, open('/Users/guidorb/GoogleDrive/SDSS/stacked/SDSSstackcatalog_stack_'+name+'_2Dadaptive_SN'+str(sn_limit)+'_'+samp+'_'+galtype+'.p', 'wb'))
	return catalog


class stack_sdss():
	def __init__(self, name, samp, galtype, xstring, ystring, save=False, adaptive=False, dimension=None, 
				plot=False, saveplot=False, 
				optional_binsize=None, datacoordinates=False, binax=None, 
				optional_xbinsize=None, optional_ybinsize=None, xdatacoordinates=False, ydatacoordinates=False,
				sort=False, sortbin=None, highsfrd=False, live=False, nstackstop=None, xbins=None, ybins=None, 
				b=None, sn_limit=None):

		if (adaptive==True) & (b==None):
			b = get_bins_2Dadaptive(samp, galtype, xstring, ystring, plot=plot, saveplot=saveplot, optional_binsize=optional_binsize, datacoordinates=datacoordinates, binax=binax)
			self.bin = b
			stack = stacking_2d_adaptive(name, samp, galtype, xstring, ystring, sn_limit=sn_limit, sortbin=sortbin, highsfrd=highsfrd, save=save, live=live, nstackstop=nstackstop, plot=plot, saveplot=saveplot, b=b, binax=binax)
		elif (adaptive==True) & (b!=None):
			stack = stacking_2d_adaptive(name, samp, galtype, xstring, ystring, sn_limit=sn_limit, sortbin=sortbin, highsfrd=highsfrd, save=save, live=live, nstackstop=nstackstop, plot=plot, saveplot=saveplot, b=b, binax=binax)
	
		elif (adaptive==False) & (xbins==None) & (ybins==None):
			xb, yb = get_bins(samp, galtype, xstring, ystring, plot=plot, saveplot=saveplot, optional_xbinsize=optional_xbinsize, optional_ybinsize=optional_ybinsize, xdatacoordinates=xdatacoordinates, ydatacoordinates=ydatacoordinates)
			self.xbin = xb
			self.ybin = yb
			if dimension=='1':
				stack = stacking_1d(name, samp, galtype, sn_limit=sn_limit, sort=sort, sortbin=sortbin, highsfrd=highsfrd, save=save, live=live, nstackstop=nstackstop)
			elif dimension=='2':
				stack = stacking_2d(name, samp, galtype, xstring, ystring, sn_limit=sn_limit, highsfrd=highsfrd, save=save, live=live, nstackstop=nstackstop, plot=plot, saveplot=saveplot, xbins=xbins, ybins=ybins)
		elif (adaptive==False) & (xbins!=None) & (ybins!=None):
			if dimension=='1':
				stack = stacking_1d(name, samp, galtype, sn_limit=sn_limit, sort=sort, sortbin=sortbin, highsfrd=highsfrd, save=save, live=live, nstackstop=nstackstop)
			elif dimension=='2':
				stack = stacking_2d(name, samp, galtype, xstring, ystring, sn_limit=sn_limit, highsfrd=highsfrd, save=save, live=live, nstackstop=nstackstop, plot=plot, saveplot=saveplot, xbins=xbins, ybins=ybins)
		
		self.catalog = stack



























