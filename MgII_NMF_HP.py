#Constants------------------------------------
first_line_wave = 2796.3543 #first line of MgII Doublet
second_line_wave = 2803.5315 #second line of MgII Doublet
rf_line_sep = second_line_wave-first_line_wave

#Doublet Finder Hyperparameters
rf_err_margain = 1.0 #acceptable difference from expected line seperation
kernel_smooth = 2 #sttdev value of gaussian smoothing kernel used in calculating residual for doublet finding step
med_filt_size_tight = 19 #narrow median filter used in constructing doublet finder continuum
med_filt_size_wide = 39 #wider median filter used in constructing doublet finder continuum

med_filt_size_fit = 85 #very wide median filter used as a fallback continuum when NMF fit is poor

snr_threshold_low = 1.5 #threshold applied to second line for doublet fit candiates
snr_threshold_high = 2.5 #threshold applied to first line for doublet fit candiates

#MCMC Hyperparamets
sub_region_size = 40 #number of pixels around line used in determining region for MCMC fit
min_accept_frac = 0.45 #minimum acceptance fraction necessary for MCMC process
min_tol = 100 #minimum chain length / max(autocorrelation time estimate) necessary for MCMC process
nwalkers = 32 #number of probability space walkers used in MCMC
ndim = 5 #number of parameters in MCMC fit (redshift, line amplitudes, line widths)
chain_length = 15000 #total MCMC step duration
chain_discard = 3000 #MCMC steps discarded from beginning of process before determining values and storing chain
#----------------------------------------------------

#multiprocessing seteup
import multiprocessing
      
#standard imports
import numpy as np
import os
import h5py
import sys
import desispec.io
import fitsio
from astropy import modeling
from argparse import ArgumentParser

import emcee #this will require some python path setup and being cloned from git

from desimodel.footprint import radec2pix
from desispec.coaddition import coadd_cameras

#doublet finder setup
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from scipy.signal import medfilt
kernel = Gaussian1DKernel(stddev = kernel_smooth)

#used in selecting groups of consecutive negative values in doublet finder
from operator import itemgetter
from itertools import *
fitter = modeling.fitting.LevMarLSQFitter()
model = modeling.models.Gaussian1D()

#used in determining QSO continuum
import NMF_quasar_SEDs_decomposition_module as NMF
NMF_basis = NMF.load_quasar_NMF_basis()

#MgII line model
def MgII_Model(theta,x):
    z,a1,a2,s1,s2 = theta
    #determine peak centers
    m1 = (z+1)*first_line_wave
    m2 = (z+1)*second_line_wave
    
    #Generate Model
    model = a1*np.exp((-(x-m1)**2)/(2*s1**2))+a2*np.exp((-(x-m2)**2)/(2*s2**2))
    return model

#basic chi2 function, used in determining continuum goodness of fit
def chi2(obs,cal,var,reduced = True):
    c2 = np.sum((obs-cal)**2 / var)
    
    if reduced:
        return c2/len(obs)
    else:
        return c2

#These next three functions used in MCMC process
#likelihood fnc
def log_likelihood(theta, x, y, yerr):
    #generate model
    model = MgII_Model(theta,x)
    #turn error into variance
    sigma2 = yerr ** 2
    #Actual Likelihood fnc
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

#prior fnc, could contain more info on reasonable redshifts, heights and widths
def log_prior(theta,z_low,z_high):
    z,a1,a2,s1,s2 = theta
    #standard deviations always positive and redshift in acceptable range
    if 0 < s1 and 0 < s2 and z_low < z < z_high:
        return 0.0
    return -np.inf

#probability fnc
def log_probability(theta, x, y, yerr, z_low, z_high):
    lp = log_prior(theta,z_low,z_high)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

#this function applies the MCMC process given an output from the doublet finder
def Doublet_MCMC(Doublet,x_spc,y_flx,y_err):
    #spectrum info
    x_spc,y_flx,y_err = x_spc,y_flx,y_err

    #determine redshift and appropriate line_sep
    z = float(Doublet[0])
    line_sep = rf_line_sep*(1+z)
    
    #new definition of peak (i.e. center of search region)
    peak = np.argmin(np.abs(x_spc - first_line_wave*(1+z)))

    #set sub region values, bounded by [0,x_spc-1]
    srh = min(len(x_spc)-1,peak+sub_region_size)
    srl = max(0,peak-sub_region_size)

    #determine max and min z in window (or lowest/highest values possible if at edge of wavelength space)
    z_low = x_spc[srl]/first_line_wave-1
    z_high = x_spc[srh]/first_line_wave-1

    #define subregion in x and y
    reg_wave = x_spc[srl:srh]
    reg_flx = y_flx[srl:srh]
    reg_err = y_err[srl:srh]

    init_Amp1 = -float(Doublet[4])
    init_Amp2 = -float(Doublet[6])

    init_StdDev1 = float(Doublet[5])
    init_StdDev2 = float(Doublet[7])

    #define initial theta (TODO: Reconsider guesses for m,b)
    initial = [z,init_Amp1,init_Amp2,init_StdDev1,init_StdDev2]

    #could widen this inital guess range, don't think it matters though
    p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)

    #setup sampler with args
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args = [reg_wave,reg_flx,reg_err,z_low,z_high])

    #burn-in
    state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    #MCMC can hit errors during running, so wrap in try/except
    try:
        state = sampler.run_mcmc(state,chain_length)
    except Exception as e:
        print(e)
        return
    
    #extract estimate autocorrelation time, number of times chain is longer than autocorr time and mean_accept_frac
    tau = sampler.get_autocorr_time(discard = chain_discard,quiet = True)
    implied_tol = chain_length/max(tau)
    mean_accept_frac = np.mean(sampler.acceptance_fraction)
    
    #mcmc quality cuts, if passed return the samples
    if(implied_tol > min_tol and mean_accept_frac > min_accept_frac):
        #extract MCMC info
        flat_samples = sampler.get_chain(flat=True,discard=chain_discard)
        return (flat_samples)
    else:
        return

#Takes spectra info and returns list of doublets that pass snr criteria
def doublet_finder(continuum,x_spc,y_flx,y_err,min_z):
    
    #calculate residual
    residual = continuum-y_flx

    #find indices where residuals are positive
    pos_inds = np.where(residual > 0)[0]
    #from: https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array, credit user: unutbu
    cons_groups = np.array(np.split(pos_inds, np.where(np.diff(pos_inds) != 1)[0]+1),dtype=object)
    #find groups longer than 2 entries
    minidx_mask = np.array([arr.size > 2 for arr in cons_groups])
    
    #check groups of 3 of more indices for snr
    absorb_lines = []
    for group in cons_groups[minidx_mask]:
        #calculate snr
        snr = np.sum(residual[group])/np.sqrt(np.sum(y_err[group]**2))
        
        #set central index to be the highest value
        cen = np.nanargmax(residual[group])
        
        #skip entries with implied redshifts below min_Z
        #could run this earlier but here seems fine
        if x_spc[group][cen]/first_line_wave - 1 < min_z:
            continue
        
        if snr > snr_threshold_low:
            try:
                model = modeling.models.Gaussian1D(amplitude = np.nanmax(residual[group]), mean = x_spc[group][cen], stddev = 0.4)
                fm = fitter(model, x = x_spc[group], y = residual[group])
            except:
                print('1D Gaussian Error',model,group)
            
            #save the line parameters as well as group of indices, snr and central index
            absorb_lines.append([fm.parameters[1],group,snr,fm.parameters[0],fm.parameters[2],cen])
    #convert to np array
    absorb_lines = np.array(absorb_lines,dtype=object)

    #calculate what implied redshift would be if each feature was first/second line of MgII doublet
    firstline_z = absorb_lines[:,0]/first_line_wave - 1
    secondline_z = absorb_lines[:,0]/second_line_wave -1
    
    #calculate the allowed redshift difference implied by the restframe seperation, restframe error margain and firstline redshift
    z_diff = ((1+firstline_z)*(first_line_wave+rf_line_sep+rf_err_margain))/second_line_wave - 1 - firstline_z

    #comparing each firstline to it's corresponding second line (note slicing) we can check if below redshift difference
    doublet_loc = np.where(secondline_z[1:]-firstline_z[:-1] < z_diff[:-1])[0]
    
    #save a doublet for each case where redshift difference is acceptable and snr thresholds are met
    doublets = []
    for loc in doublet_loc:
        
        line1 = absorb_lines[loc]
        line2 = absorb_lines[loc+1]
        
        #check if leading line (2796) meets high snr threshold
        if line1[2] > snr_threshold_high:
            #append the redshift, center index, snrs and initial line amplitudes and widths
            doublets.append([firstline_z[loc],line1[5],line1[2],line2[2],line1[3],line1[4],line2[3],line2[4]])
        
    return np.array(doublets)

def Doublet_Detection(cat_subset,hp):
    #for the healpixel determine the approriate prefix pixel (i.e. the directory structure)
    hp_str = str(hp)
    if int(hp) < 100:
        hp_short = '0'
    else:
        hp_short = hp_str[0:-2]

    surveys = np.unique(cat_subset['SURVEY'])

    #format output directory and filename
    out_dir = '{}/{}/{}'.format(out_base_dir,hp_short,hp_str)
    print('Output Directory:', out_dir)
    out_fn = 'MgII-Abs-Chains-{}.hdf5'.format(hp_str)
    #names of model params for h5py formatting
    names = ['z','Amp1','Amp2','StdDev1','StdDev2']

    #if output file path does not exist create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #full format out filepath
    out_file = os.path.join(out_dir,out_fn)

    #open h5py buffer to write out file
    with h5py.File(out_file, "w") as f:
        for survey in surveys:
            #format coadd directory and filename
            in_dir = os.path.join(reduction_base_dir, "healpix",survey,"dark",hp_short,hp_str)
            in_fn = "coadd-{}-{}-{}.fits".format(survey, "dark", hp)

            #read in specobj file, and also coadd
            specfile = os.path.join(in_dir, in_fn)
            #print(specfile)
            specobj = desispec.io.read_spectra(specfile)
            coadd_specobj = coadd_cameras(specobj)
            
            print('Running healpixel: {} with {} QSO catalog entries'.format(hp_str,coadd_specobj.num_spectra()))

            #grab wavelength grid as this will be same for all spectra
            x_spc = coadd_specobj.wave["brz"]
            
            cat_srv_sub = cat_subset[cat_subset['SURVEY']==survey]

            #find indices of specobj that have a targetid value in the catalog subset
            #i.e. find the indices of the QSO_cat entries in the specobj
            #QSO_inds = np.where(np.isin(specobj.target_ids(),cat_subset['TARGETID'])==True)[0]

            #iterate over qso indices and the related shift from catalog subset
            #little opaque here but clean
            for entry in cat_srv_sub:
                
                redshift = entry['Z']
                TARGETID = entry['TARGETID']
                
                #have to pull out from inside listed list
                ind = np.where(specobj.target_ids() == TARGETID)[0][0]
 

                #calculate wavelength of MgII abs at Lya emission (we don't search this region)
                min_z = (1216*(1+redshift)/first_line_wave)-1

                #grab flux and error date for specific spectra
                y_flx = coadd_specobj.flux["brz"][ind]
                y_err = 1/np.sqrt(coadd_specobj.ivar["brz"][ind])
                
                #apply gaussian smoothing kernel
                smooth_yflx = convolve(y_flx,kernel)
                #estimate continuum using combination of two median filters
                c_tight = medfilt(y_flx,med_filt_size_tight)
                c_wide = medfilt(y_flx,med_filt_size_wide)
                #length of filter scale for normalizing
                f_scale = len(c_tight)
                #weight tight filter higher at low z and wide filter higher at high z
                doublet_cont = c_tight*(1-np.arange(f_scale )/f_scale) + c_wide*(np.arange(f_scale )/f_scale)
                #run doublet finder
                doublets = doublet_finder(doublet_cont,x_spc,smooth_yflx,y_err,min_z)

                #NEW code for NMF/medfilt continuum conditions
                NMF_fail = False
                Cont_method = ''
                #using NMF estimator, will sometimes fail so t/e wrapped. If it fails we want to try medfilt continuum so set bool
                try:
                    out = NMF.NMF_normalization_v2(np.log10(x_spc),y_flx,coadd_specobj.ivar["brz"][ind],redshift,NMF_basis)
                except Exception as e:
                    print(e)
                    NMF_fail = True
                
                #if NMF failed, always use medfilt continuum
                if NMF_fail:
                    
                    #calculate new medilt and set as fitting_cont, set cont_method
                    medfilt_cont = medfilt(y_flx,med_filt_size_fit)
                    fitting_cont = medfilt_cont
                    Cont_method = 'Medfilt'
                
                #if NMF is poorly fit determine fitting continuum based on chi2 values
                elif out[-1] > 4.0:
                    
                    #calculate new medilt and chi2
                    medfilt_cont = medfilt(y_flx,med_filt_size_fit)
                    medfilt_chi2 = chi2(y_flx,medfilt_cont,coadd_specobj.ivar["brz"][ind])
                    
                    #choose continuum with better chi2
                    if medfilt_chi2 < out[-1]:
                        fitting_cont = medfilt_cont
                        Cont_method = 'Medfilt'
                    else:
                        fitting_cont = out[-2]
                        Cont_method = 'NMF'
                
                else:
                    fitting_cont = out[-2]
                    Cont_method = 'NMF'

                #run each doublet candidate through MCMC process
                for doublet in doublets:
                    chain = Doublet_MCMC(doublet,x_spc,y_flx-fitting_cont,y_err)

                    #if the chain returned is actually an object (i.e. MCMC didn't crash)
                    if not chain is None:
                        #write hp name for entry, with MgII Abs redshift and QSO em redshift
                        z_MgII = np.round(np.percentile(chain[:, 0],50),decimals = 5)
                        z_QSO = np.round(redshift,decimals=5)
                        out_str = '{}_{}_{}_{}_{}_{}_{}_{}'.format(TARGETID,survey,hp,z_MgII,z_QSO,doublet[2],doublet[3],Cont_method)
                        #create h5py group, TODO: way to check if out_str is an h5py group w/o try/except?
                        try:
                            grp = f.create_group(out_str)
                        except ValueError as e:
                            #if group already exists then this MgII feature is already recorded, move to next entry
                            print('Error in storing h5py',e)
                            continue
                        for k in range(ndim):
                            grp.create_dataset(names[k],dtype = float,data = chain[:,k])

#Function pool will pass each hp_unique entry to
def RunFinder_HPind(hp_val):
    #pull subset of QSO_cat in healpix area
    cat_sub = QSOcat[hp_vals==hp_val]
    #run feature detection and MCMC step
    Doublet_Detection(cat_sub,hp_val)
    print('Done: ',hp_val)

def parse():
    parser = ArgumentParser(description="This script produces hdf5 files with MgII absorber information for all spectra in a specific spectroscopic release.")
    parser.add_argument("--redux", help = "Spectroscopic reduction to search for MgII Absorbers",type=str,default='fuji')
    parser.add_argument("--redux_split", help = "Used to divide the reduction into multiple portions to be run across multiple nodes", type=int, default=1)
    parser.add_argument("--split_num", help = "When redux_split > 1 this should be specified to determine which portion of the split to run", type=int, default=None)
    parser.add_argument("--num_proc", help = "Number of processes to use, recommend 1 to 32 to avoid memory issues. When num_proc = 1 only a single healpixel will be run",type=int,default=1)
    parser.add_argument("--outdir",help = "Directory to write results of MgII search. Subdirectories MgII_Abs_Chains/{redux} will be created.",type=str, default=os.environ['PSCRATCH'])
    
    args = parser.parse_args()
    return args

def main():
    args = parse()

    #These pased as command line args
    redux = args.redux
    global reduction_base_dir; global out_base_dir
    reduction_base_dir = '/global/cfs/cdirs/desi/spectro/redux/{}/'.format(redux)
    out_base_dir = os.path.join(args.outdir,'MgII_Abs_Chains/{}'.format(redux))
    
    #determine catalog path from redux. Requires adding a new entry for each spectroscopic release
    if redux == 'fuji' or redux == 'guadalupe':  
        QSOcat_fp = '/global/cfs/cdirs/desi/users/edmondc/QSO_catalog/{}/QSO_cat_{}_healpix_only_qso_targets.fits'.format(redux,redux)        
    elif redux == 'iron':
        QSOcat_fp = '/global/cfs/cdirs/desi/survey/catalogs/Y1/QSO/iron/QSO_cat_iron_main_dark_healpix_only_qso_targets_v0.fits'
    elif redux == 'loa':
        QSOcat_fp = '/global/cfs/cdirs/desi/survey/catalogs/DA2/QSO/loa/QSO_cat_loa_main_dark_healpix_only_qso_targets_v2.fits'
    else:
        print('Specified reduction:{} has no provided QSO catalog filepath'.format(redux))
        sys.exit()

    global QSOcat
    QSOcat = fitsio.read(QSOcat_fp,'QSO_CAT')
    #restrict to only dark time exposures
    QSOcat = QSOcat[QSOcat['PROGRAM'] == 'dark']
        
    print('{} entries in input QSO Catalog'.format(QSOcat.size))
    
    #create list of unique healpix values
    global hp_vals
    hp_vals = radec2pix(64,QSOcat['TARGET_RA'],QSOcat['TARGET_DEC'])
    hp_unique = np.unique(hp_vals)
    
    #routine to check if hp values have been completed
    #make a mask of healpix directories that have/have not been succesfully complete    
    hp_complete_mask = []
    
    for hp_val in hp_unique:
        #recast as str
        hp_str = str(hp_val)
        
        if int(hp_val) < 100:
            hp_short = 0
        else:
            hp_short = hp_str[0:-2]
    
        #format filepath to MgII output
        out_dir = '{}/{}/{}'.format(out_base_dir,hp_short,hp_str)
        h5_fn = 'MgII-Abs-Chains-{}.hdf5'.format(hp_str)
        h5_fp = os.path.join(out_dir,h5_fn)
    
        #see if a valid h5 file is written into path (TODO: a way to do this without try/except?)
        try:
            f = h5py.File(h5_fp,'r')
            #if it is (i.e. no error thrown) we have completed this hp_val (mask[val]=True)
            hp_complete_mask.append(True)
        #if a valid h5 file hasn't been written. we can check if it exists and remove it, then proceed. Either way (mask[val]=False)
        except OSError as e:
            if(str(e)[0:19] == 'Unable to open file' and os.path.exists(h5_fp)):
                os.remove(h5_fp)
            hp_complete_mask.append(False)
    
    #np-ify (TODO: better way to do this?)
    hp_complete_mask = np.asarray(hp_complete_mask)
    
    #find subset of incomplete hp_dirs
    hp_incomplete = hp_unique[~hp_complete_mask]
    print('{} total healpixels in release'.format(len(hp_unique)))
    print('{} incomplete healpixels in release'.format(len(hp_incomplete)))
    
    #check for args redux split and divide by split number if necessary
    if args.redux_split > 1:
        start_pct = int(100*(args.split_num-1)/args.redux_split)
        end_pct = int(100*args.split_num/args.redux_split)

        hp_incomplete = hp_incomplete[int(hp_incomplete.size*start_pct/100):int(hp_incomplete.size*end_pct/100)]
        print('Split {} contains {} healpixels to run'.format(args.split_num,hp_incomplete.size))
    
    print('Running MgII Absorption Finder on healpix directories: {}.'.format(hp_incomplete))
    
    #If there are no incomplete directories, survey is complete. Exit
    if(len(hp_incomplete) == 0):
        print('All healpix directories are already completed for release: {}'.format(redux))
        sys.exit()
          
    if args.num_proc == 1:
        #run a single random entry with a single process
        nprocesses = 1
        print('Running in single process safe mode')
        RunFinder_HPind(hp_incomplete[0])
    
    else:
        #Setup and run pool
        pool = multiprocessing.Pool(processes = args.num_proc)
        print('{} processes being utilized'.format(args.num_proc))
        #run pool
        pool.map(RunFinder_HPind, hp_incomplete)

main()
