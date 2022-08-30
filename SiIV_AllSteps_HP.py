#Constants------------------------------------
first_line_wave=1393.75
second_line_wave=1402.77
rf_line_sep=second_line_wave-first_line_wave

#Doublet Finder Hyperparameters
rf_err_margain=0.50
kernel_smooth=2
med_filt_size=39

snr_threshold_low=1.0
snr_threshold_high=2.0

#MCMC Hyperparameters
sub_region_size=50
min_accept_frac=0.45
min_tol=100
nwalkers = 32
ndim=7 #not really a hyper
chain_length=20000
chain_discard=2000
#----------------------------------------------------

#multiprocessing seteup
import multiprocessing
nprocesses = multiprocessing.cpu_count() // 4
      
#standard imports
import numpy as np
import os
import h5py
import sys
import desispec.io
import fitsio

from astropy import modeling
#from astropy.table import Table
from scipy.signal import medfilt

import emcee

from desimodel.footprint import radec2pix
from desispec.coaddition import coadd_cameras

#doublet finder setup
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
kernel = Gaussian1DKernel(stddev=kernel_smooth)

from operator import itemgetter
from itertools import *
fitter = modeling.fitting.LevMarLSQFitter()
model = modeling.models.Gaussian1D()

#using the main and sv cmx
from desitarget.cmx.cmx_targetmask import cmx_mask
from desitarget.targets import main_cmx_or_sv

def SiIV_Model(theta,x):
    z,a1,a2,s1,s2,m,b=theta
    #determine peak centers
    m1=(z+1)*first_line_wave
    m2=(z+1)*second_line_wave
    
    #Generate Model
    model = m*(x-x[0]) +b + a1*np.exp((-(x-m1)**2)/(2*s1**2))+a2*np.exp((-(x-m2)**2)/(2*s2**2))
    return model

#likelihood fnc
def log_likelihood(theta, x, y, yerr):
    #generative model
    model = SiIV_Model(theta,x)
    #error into variance
    sigma2 = yerr ** 2
    #Actual Likelihood fnc
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

#prior fnc, could contain more info on reasonable redshifts, heights and widths
def log_prior(theta,z_low,z_high):
    z,a1,a2,s1,s2,m,b=theta
    #if -100 < a1 < 100 and  -100 < a2 < 100 and 0 < s1  and  0 < s2  and z_low < z < z_high:
    if 0 < s1 and 0 < s2 and z_low < z < z_high:
        return 0.0
    return -np.inf

#probability fnc
def log_probability(theta, x, y, yerr, z_low, z_high):
    lp = log_prior(theta,z_low,z_high)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

def Doublet_MCMC(Doublet,x_spc,y_flx,y_err):
    #spectrum info
    x_spc,y_flx,y_err=x_spc,y_flx,y_err

    #determine redshift and appropriate line_sep
    z=float(Doublet[0])
    line_sep=rf_line_sep*(1+z)
    peak=int(Doublet[1])

    #set sub region values, bounded by [0,x_spc-1]
    srh=min(len(x_spc)-1,peak+sub_region_size)
    srl=max(0,peak-sub_region_size)

    #determine max and min z in window (or lowest/highest values possible if at edge of wavelength space)
    z_low=x_spc[srl]/first_line_wave-1
    z_high=x_spc[srh]/first_line_wave-1

    #define subregion in x and y
    reg_wave=x_spc[srl:srh]
    reg_flx=y_flx[srl:srh]
    reg_err=y_err[srl:srh]

    #initial line guesses
    init_m=(reg_flx[0]-reg_flx[-1])/(reg_wave[0]-reg_wave[-1])
    init_b=reg_flx[0]

    init_Amp1= -float(Doublet[4])
    init_Amp2= -float(Doublet[6])

    init_StdDev1= float(Doublet[5])
    init_StdDev2= float(Doublet[7])

    #define initial theta (TODO: Reconsider guesses for m,b)
    initial=[z,init_Amp1,init_Amp2,init_StdDev1,init_StdDev2,init_m,init_b]

    #could widen this inital guess range, don't think it matters though
    p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)

    #setup sampler with args
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[reg_wave,reg_flx,reg_err,z_low,z_high])

    #burn-in
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    #MCMC can hit errors during running, so wrap in try/except
    try:
        state=sampler.run_mcmc(state,chain_length)
    except ValueError:
        return
    
    #extract estimate autocorrelation time, number of times chain is longer than autocorr time and mean_accept_frac
    tau = sampler.get_autocorr_time(discard=chain_discard,quiet=True)
    implied_tol=chain_length/max(tau)
    mean_accept_frac=np.mean(sampler.acceptance_fraction)
    
    
    #mcmc quality cuts, if passed return the samples
    if(implied_tol > min_tol and mean_accept_frac > min_accept_frac):
        #extract MCMC info
        flat_samples = sampler.get_chain(flat=True,discard=chain_discard)
        return (flat_samples)
    else:
        return

#Takes spectra info and returns list of doublets that pass snr criteria
def doublet_finder(continuum,x_spc,y_flx,y_err,min_z):
    
    residual=continuum-y_flx
    #Generate groups of data with positive residuals
    #From https://stackoverflow.com/questions/3149440/python-splitting-list-based-on-missing-numbers-in-a-sequence
    groups = []
    for k, g in groupby(enumerate(np.where(residual>0)[0]), lambda x: x[0]-x[1]):
        groups.append(list(map(itemgetter(1), g)))
    absorb_lines=[]
    
    for group in groups:
        #Skip groups of 1 or 2 data vals, these aren't worthwhile peaks and cause an issue in fitting
        if(len(group) < 3):
            continue

        snr=np.sum(residual[group])/np.sqrt(np.sum(y_err[group]**2))
        
        #HYPERPARAMETER
        if(snr>snr_threshold_low):
            cen=int(np.average(group))
            #Fit gaussian model
            model = modeling.models.Gaussian1D(amplitude=np.nanmax(residual[group]),mean=np.average(x_spc[group]))
            fm = fitter(model=model, x=x_spc[group], y=residual[group])
            #determine redshift by model params
            
            absorb_lines.append([fm.parameters[1],group,snr,fm.parameters[0],fm.parameters[2],cen])
    
    doublets=[]
    #This is super poorly optimized, but just not too much of a slowdown
    #should only look at line2s at higher wavelength than line1
    for line1 in absorb_lines:
        z=line1[0]/first_line_wave-1
        line_sep=rf_line_sep*(1+z)
        err_margain=rf_err_margain*(1+z)

        for line2 in absorb_lines:
            #if wavelength range is good, redshift is past LyA and one of the line snrs is above snr_threshold_high
            if(line1[0]+line_sep-err_margain<line2[0]<line1[0]+line_sep+err_margain and z>min_z and max(line1[2],line2[2])>snr_threshold_high):
                #pass along the redshift of first line, center value of first line, snr of both lines, amplitude and std-dev of both lines
                doublets.append([z,line1[5],line1[2],line2[2],line1[3],line1[4],line2[3],line2[4]])
    return doublets

def Doublet_Detection(cat_subset,hp):
    hp_str=str(hp)

    surveys=np.unique(cat_subset['SURVEY'])

    #format output directory and filename
    out_dir='/global/cscratch1/sd/lucasnap/SiIV_Abs_Chains/{}/{}/{}'.format(redux,hp_str[0:-2],hp_str)
    out_fn='SiIV-Abs-Chains-{}.hdf5'.format(hp_str)
    #names of model params for h5py formatting
    names=['z','Amp1','Amp2','StdDev1','StdDev2','m','b']

    #if output file path does not exist create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #full format out filepath
    out_file = os.path.join(out_dir,out_fn)

    #open h5py buffer to write out file
    with h5py.File(out_file, "w") as f:
        for survey in surveys:
            #format coadd directory and filename
            in_dir = os.path.join(reduction_base_dir, "healpix",survey,"dark",hp_str[0:-2],hp_str)
            in_fn = "coadd-{}-{}-{}.fits".format(survey, "dark", hp)

            #read in specobj file, and also coadd
            specfile = os.path.join(in_dir, in_fn)
            specobj = desispec.io.read_spectra(specfile)
            coadd_specobj=coadd_cameras(specobj)

            #grab wavelength grid as this will be same for all spectra
            x_spc=coadd_specobj.wave["brz"]

            #find indices of specobj that have a targetid value in the catalog subset
            #i.e. find the indices of the QSO_cat entries in the specobj
            QSO_inds=np.where(np.isin(specobj.target_ids(),cat_subset['TARGETID'])==True)[0]

            #iterate over qso indices and the related shift from catalog subset
            #little opaque here but clean
            for ind,redshift in zip(QSO_inds,cat_subset['Z']):
                #pull targetid
                TARGETID=specobj.target_ids()[ind]

                #calculate wavelength of SiIV abs at Lya emission (we don't search this region)
                min_z=(1216*(1+redshift)/first_line_wave)-1

                #grab flux and error date for specific spectra
                y_flx = coadd_specobj.flux["brz"][ind]
                y_err=1/np.sqrt(coadd_specobj.ivar["brz"][ind])


                #apply gaussian smoothing kernel
                smooth_yflx=convolve(y_flx,kernel)
                #estimate continuum using median filter
                cont_est = medfilt(y_flx,med_filt_size)
                #run doublet finder
                doublets=doublet_finder(cont_est,x_spc,smooth_yflx,y_err,min_z)


                for doublet in doublets:
                    chain=Doublet_MCMC(doublet,x_spc,y_flx,y_err)

                    #if the chain returned is actually an object (i.e. MCMC didn't crash)
                    if not chain is None:
                        #write hp name for entry, with SiIV Abs redshift and QSO em redshift
                        z_SiIV=np.round(np.percentile(chain[:, 0],50),decimals=5)
                        z_QSO=np.round(redshift,decimals=5)
                        out_str='{}_{}_{}_{}_{}_{}_{}'.format(TARGETID,survey,hp,z_SiIV,z_QSO,doublet[2],doublet[3])
                        #create h5py group, TODO: way to check if out_str is an h5py group w/o try/except?
                        try:
                            grp = f.create_group(out_str)
                        except ValueError as e:
                            #if group already exists then this SiIV feature is already recorded, move to next entry
                            continue
                        for k in range(ndim):
                            grp.create_dataset(names[k],dtype=float,data=chain[:,k])

#These pased as command line args
redux=str(sys.argv[1])
reduction_base_dir='/global/cfs/cdirs/desi/spectro/redux/{}/'.format(redux)

#open catalog
QSOcat_fp='/global/cfs/cdirs/desi/users/edmondc/QSO_catalog/{}/QSO_cat_{}_healpix.fits'.format(redux,redux)
QSOcat=fitsio.read(QSOcat_fp,'QSO_CAT')

#restrict to only dark time exposures (??)
QSOcat = QSOcat[QSOcat['PROGRAM'] == 'dark']

#create list of unique healpix values
hp_vals=radec2pix(64,QSOcat['TARGET_RA'],QSOcat['TARGET_DEC'])
hp_unique=np.unique(hp_vals)

#routine to check if hp values have been completed
#make a mask of healpix directories that have/have not been succesfully complete    
hp_complete_mask=[]

for hp_val in hp_unique:
    #recast as str
    hp_str=str(hp_val)

    #format filepath to SiIV output
    out_dir='/global/cscratch1/sd/lucasnap/SiIV_Abs_Chains/{}/{}/{}'.format(redux,hp_str[0:-2],hp_str)
    h5_fn='SiIV-Abs-Chains-{}.hdf5'.format(hp_str)
    h5_fp=os.path.join(out_dir,h5_fn)

    #see if a valid h5 file is written into path (TODO: a way to do this without try/except?)
    try:
        f=h5py.File(h5_fp,'r')
        #if it is (i.e. no error thrown) we have completed this hp_val (mask[val]=True)
        hp_complete_mask.append(True)
    #if a valid h5 file hasn't been written. we can check if it exists and remove it, then proceed. Either way (mask[val]=False)
    except OSError as e:
        if(str(e)[0:19]=='Unable to open file' and os.path.exists(h5_fp)):
            os.remove(h5_fp)
        hp_complete_mask.append(False)

#np-ify (TODO: better way to do this?)
hp_complete_mask=np.asarray(hp_complete_mask)

#find subset of incomplete hp_dirs
hp_incomplete=hp_unique[~hp_complete_mask]

#take percentage of hp_incomplete corresponding to passed argv
#currently defeaults to 10 percent interval
hp_incomplete=hp_incomplete[int(len(hp_incomplete)*int(sys.argv[3])/100):int(len(hp_incomplete)*(int(sys.argv[3])/100+0.1))]

print('Running SiIV Absorption Finder on healpix directories: {}.'.format(hp_incomplete))

#If there are no incomplete directories, survey is complete. Exit
if(len(hp_incomplete)==0):
    print('All healpix directories are already completed for release: {}'.format(redux))
    sys.exit()

#Function pool will pass each hp_unique entry to
def RunFinder_HPind(hp_val):
            
    #pull subset of QSO_cat in healpix area
    cat_sub=QSOcat[hp_vals==hp_val]
    #run feature detection and MCMC step
    Doublet_Detection(cat_sub,hp_val)
    print('Done: ',hp_val)

    
if (str(sys.argv[2])=='safe'):
    #run a single random entry with a single process
    nprocesses=1
    print('{} processes being utilized'.format(nprocesses))
    #RunFinder_HPind(np.random.choice(hp_incomplete))
    
    RunFinder_HPind(hp_incomplete[0])
    RunFinder_HPind(hp_incomplete[1])

if (str(sys.argv[2])=='full'):
    #Setup and run pool
    pool = multiprocessing.Pool(processes=nprocesses)
    print('{} processes being utilized'.format(nprocesses))
    #run pool
    pool.map(RunFinder_HPind, hp_incomplete)