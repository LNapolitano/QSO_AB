#!/usr/bin/env python
# coding: utf-8

# Code designed to write catalogs of absorbers from .h5py files stored within each healpix directory. Then individual healpix catalogs are merged into a single release based file.

# In[1]:


import numpy as np
import h5py
import os
import sys

import desispec.io
import fitsio
from desispec.coaddition import coadd_cameras
from desimodel.footprint import radec2pix

import multiprocessing
nprocesses = multiprocessing.cpu_count() // 8

from scipy.signal import medfilt
med_filt_size = 85

#Example chains_fps
#'/pscratch/sd/l/lucasnap/MgII_Abs_Chains/fuji-NMF'
#'/global/cscratch1/sd/lucasnap/SiIV_Abs_Chains'

#file path setup for SiIV chains
chains_fp = str(sys.argv[1])

#split filepath to get reduction and which_line catalogs are being written for
chain_spt = chains_fp.split('/')

redux = chain_spt[-1].split('-')[0]

which_line = chain_spt[-2].split('_')[0]

if which_line == 'MgII':
    first_line_wave = 2796.3543
    second_line_wave = 2803.5315
    line = 'MGII'
elif which_line == 'SiIV':
    first_line_wave = 1393.75
    second_line_wave = 1402.77
    line = 'SIIV'
else:
    print('Invalid command line arg')
    sys.exit()


center = (second_line_wave+first_line_wave)/2

ndim = 5

#units of columns in resulting catalogs
Catalog_units = ['',"deg","deg",'','','','','','','angstrom','angstrom','angstrom','angstrom','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','','']

#names of columns in resulting catalogs
names =['TARGETID','RA','DEC','SURVEY','ZWARN','TSNR2_QSO','TSNR2_LYA','TSNR2_LRG','Z_QSO','EW_2796','EW_2803','EW_2796_ERR_LOW','EW_2803_ERR_LOW','EW_2796_ERR_HIGH','EW_2803_ERR_HIGH','Z_{}'.format(line),'AMP_2796','AMP_2803','STDDEV_2796','STDDEV_2803','Z_{}_ERR_LOW'.format(line),'AMP_2796_ERR_LOW','AMP_2803_ERR_LOW','STDDEV_2796_ERR_LOW','STDDEV_2803_ERR_LOW','Z_{}_ERR_HIGH'.format(line),'AMP_2796_ERR_HIGH','AMP_2803_ERR_HIGH','STDDEV_2796_ERR_HIGH','STDDEV_2803_ERR_HIGH','CONTINUUM_METHOD','LINE_SNR_MIN','LINE_SNR_MAX']

#file path setup for reduction directories
reduction_base_dir = '/global/cfs/cdirs/desi/spectro/redux/{}/'.format(redux)
summary_cat_dir = os.path.join(reduction_base_dir,'zcatalog')

#reading all survey zcats and forming a dictionary to easily access when needed
if(redux == 'fuji'):
    sv1_zcat = fitsio.read('{}/zpix-{}-dark.fits'.format(summary_cat_dir,'sv1'),'ZCATALOG')
    sv2_zcat = fitsio.read('{}/zpix-{}-dark.fits'.format(summary_cat_dir,'sv2'),'ZCATALOG')
    sv3_zcat = fitsio.read('{}/zpix-{}-dark.fits'.format(summary_cat_dir,'sv3'),'ZCATALOG')
    
    zcat_dic = {'sv1':sv1_zcat,'sv2':sv2_zcat,'sv3':sv3_zcat}
    
else:
    main_zcat = fitsio.read('{}/zpix-{}-dark.fits'.format(summary_cat_dir,'main'),'ZCATALOG')
    
    zcat_dic = {'main':main_zcat}

#reading QSOcat
#open catalog
if redux == 'fuji' or redux == 'guadalupe':
    QSOcat_fp = '/global/cfs/cdirs/desi/users/edmondc/QSO_catalog/{}/QSO_cat_{}_healpix_only_qso_targets.fits'.format(redux,redux)
elif redux == 'iron':
    QSOcat_fp = '/global/cfs/cdirs/desi/survey/catalogs/Y1/QSO/{}/QSO_cat_{}_main_dark_healpix_only_qso_targets_vtest.fits'.format(redux,redux)
QSOcat = fitsio.read(QSOcat_fp,'QSO_CAT')


#lists to store the healpix directories with/without written .fits SiIV_catalogs
hp_complete = []
hp_incomplete = []

for short_hp in os.listdir(chains_fp):

    if short_hp[0] == '.':
        continue

    short_fp=os.path.join(chains_fp,short_hp)


    #check if filepath is a directory
    #if not move to next entry
    if not os.path.isdir(short_fp):
        continue

    for healpix in os.listdir(short_fp):
        
        if healpix[0] == '.':
            continue

        #output prep, removing file if it exists to avoid appending rather than overwriting
        out_fn='{}/{}-Absorbers-{}.fits'.format(os.path.join(chains_fp,short_hp,healpix),which_line,healpix)


        #check if out-file exists and if it does remove it
        if os.path.exists(out_fn):
            hp_complete.append(healpix)
            continue
        else:
            hp_incomplete.append(healpix)
                
                
print('{} hp catalogs complete'.format(len(hp_complete)))
print('{} hp catalogs incomplete'.format(len(hp_incomplete)))


if(redux=='fuji'):
    possible_srv=['sv1','sv2','sv3']
else:
    possible_srv=['main']

#for healpix in hp_complete:
    
def write_hp_catalog(healpix):

    #healpixel abbreviation
    if int(healpix) < 100:
        short_hp = '0'
    else:
        short_hp = healpix[0:-2]

    #output prep, removing file if it exists to avoid appending rather than overwriting
    out_fn='{}/{}-Absorbers-{}.fits'.format(os.path.join(chains_fp,short_hp,healpix),which_line,healpix)

    #go grab spectra file
    coadd_dic = {}
    for survey in possible_srv:
        hp_dir = os.path.join(reduction_base_dir,'healpix',survey,'dark',short_hp,healpix)
        specfile = os.path.join(hp_dir,'coadd-{}-dark-{}.fits'.format(survey,healpix))

        if os.path.exists(specfile):
            specobj = desispec.io.read_spectra(specfile)

            coadd_dic[survey] = specobj


    #Stupid dumb way to hold lists of values
    TARGETIDs = []
    RAs = []
    DECs = []
    Surveys = []
    ZWarns = []
    TSNR_QSOs = []
    TSNR_LYAs = []
    TSNR_LRGs = []
    z_QSOs = []

    EW_2796s_cen = []; EW_2796s_low = []; EW_2796s_high = []
    EW_2803s_cen = []; EW_2803s_low = []; EW_2803s_high = []

    z_ABS_cen = []; z_ABS_low = []; z_ABS_high = []
    Amp1s_cen = []; Amp1s_low = []; Amp1s_high = []
    Amp2s_cen = []; Amp2s_low = []; Amp2s_high = []
    StdDev1s_cen = []; StdDev1s_low = []; StdDev1s_high = []
    StdDev2s_cen = []; StdDev2s_low = []; StdDev2s_high = []

    LineSNRmins = []
    LineSNRmaxs = []
    
    ContinuumMethods = []



    h5_fn = '{}-Abs-Chains-{}.hdf5'.format(which_line,healpix)
    h5_fp = os.path.join(chains_fp,short_hp,healpix,h5_fn)
    try:
        h5_file = h5py.File(h5_fp,'r')
    except:
        print('Broken file: {}'.format(h5_fp))
        return

    #print(h5_file.keys())
    for key in h5_file.keys():

        TARGETID,survey,hp,z_ABS,z_QSO,snr1,snr2,contmeth = key.split('_')

        #excluding special survey at the moment
        if(survey == 'special'):
            continue

        #grab correct summary catalog and coadd file
        coadd = coadd_dic[survey]
        summary_cat = zcat_dic[survey]
        #find entry in summary catalog
        sum_cat_idx = np.where(summary_cat['TARGETID']==int(TARGETID))[0]
        specobj_idx = np.where(coadd.target_ids() == int(TARGETID))[0]

        #grab correct spectra and coadd (over cameras)
        spec = coadd_cameras(coadd[specobj_idx])

        #grab flux values
        x_spc = spec.wave["brz"]
        y_flx = spec.flux["brz"][0]

        #estimate continuum using median filter
        cont_est = medfilt(y_flx,med_filt_size)

        #find correct index for center of doublet
        index = np.abs(x_spc - (center*(1+float(z_ABS)))).argmin()
        #estimate continuum at this center
        continuum = float(cont_est[index])

        #en["NEW_EW2796"]=np.abs(en['AMP_2796'])*en['STDDEV_2796']*np.sqrt(2*np.pi)/(1.0+en['Z_SiIV'])/continuum
        #en["NEW_EW2803"]=np.abs(en['AMP_2803'])*en['STDDEV_2803']*np.sqrt(2*np.pi)/(1.0+en['Z_SiIV'])/continuum

        #pull info from summary catalog, ra, dec, rmag
        ra = summary_cat['TARGET_RA'][sum_cat_idx]
        dec = summary_cat['TARGET_DEC'][sum_cat_idx]

        ZWarn = summary_cat['ZWARN'][sum_cat_idx]

        TSNR_QSO = summary_cat['TSNR2_QSO'][sum_cat_idx]
        TSNR_LYA = summary_cat['TSNR2_LYA'][sum_cat_idx]
        TSNR_LRG = summary_cat['TSNR2_LRG'][sum_cat_idx]

        #RFlux=summary_cat['FLUX_R'][sum_cat_idx]

        #grab datasets assocaited with key
        dset = h5_file[key]
        #grab datasets keys
        dkeys = dset.keys()

        #hold attributes of SiIV abs. fit
        feature_mid = np.zeros(ndim)
        feature_low = np.zeros(ndim)
        feature_high = np.zeros(ndim)

        for i,dkey in zip(range(ndim),dkeys):
            feature_mid[i] = np.percentile(dset[dkey],50)
            feature_low[i] = np.percentile(dset[dkey],16)
            feature_high[i] = np.percentile(dset[dkey],84)


        #calculate line equivalent widths using amplitudes, widths, redshift and line intercept* (TODO: evaluate this choice?)
        EW_2796s = np.abs(np.random.choice(dset['Amp1'],10000))*np.random.choice(dset['StdDev1'],10000)*np.sqrt(2*np.pi)            /(1+np.random.choice(dset['z'],10000))/continuum

        EW_2803s = np.abs(np.random.choice(dset['Amp2'],10000))*np.random.choice(dset['StdDev2'],10000)*np.sqrt(2*np.pi)            /(1+np.random.choice(dset['z'],10000))/continuum

        #append everything to the dumb arrays
        TARGETIDs.append(int(TARGETID))
        RAs.append(ra)
        DECs.append(dec)
        Surveys.append(survey)
        ZWarns.append(ZWarn)
        TSNR_QSOs.append(TSNR_QSO)
        TSNR_LYAs.append(TSNR_LYA)
        TSNR_LRGs.append(TSNR_LRG)
        z_QSOs.append(float(z_QSO))

        EW_2796s_cen.append(np.percentile(EW_2796s,50))
        EW_2796s_low.append(np.percentile(EW_2796s,50)-np.percentile(EW_2796s,16))
        EW_2796s_high.append(np.percentile(EW_2796s,84)-np.percentile(EW_2796s,50))

        EW_2803s_cen.append(np.percentile(EW_2803s,50))
        EW_2803s_low.append(np.percentile(EW_2803s,50)-np.percentile(EW_2803s,16))
        EW_2803s_high.append(np.percentile(EW_2803s,84)-np.percentile(EW_2803s,50))
        

        z_ABS_cen.append(feature_mid[4]); z_ABS_low.append(feature_mid[4]-feature_low[4]); z_ABS_high.append(feature_high[4]-feature_mid[4])
        Amp1s_cen.append(feature_mid[0]); Amp1s_low.append(feature_mid[0]-feature_low[0]); Amp1s_high.append(feature_high[0]-feature_mid[0])
        Amp2s_cen.append(feature_mid[1]); Amp2s_low.append(feature_mid[1]-feature_low[1]); Amp2s_high.append(feature_high[1]-feature_mid[1])
        StdDev1s_cen.append(feature_mid[2]); StdDev1s_low.append(feature_mid[2]-feature_low[2]); StdDev1s_high.append(feature_high[2]-feature_mid[2])
        StdDev2s_cen.append(feature_mid[3]); StdDev2s_low.append(feature_mid[3]-feature_low[3]); StdDev2s_high.append(feature_high[3]-feature_mid[3])

        LineSNRmins.append(np.min([float(snr1),float(snr2)]))
        LineSNRmaxs.append(np.max([float(snr1),float(snr2)]))
        
        ContinuumMethods.append(contmeth)

    TARGETIDs = np.asarray(TARGETIDs)
    RAs = np.asarray(RAs)
    DECs = np.asarray(DECs)
    Surveys = np.asarray(Surveys)
    ZWarns = np.asarray(ZWarns)      
    TSNR_QSOs = np.asarray(TSNR_QSOs)
    TSNR_LYAs = np.asarray(TSNR_LYAs)
    TSNR_LRGs = np.asarray(TSNR_LRGs)  
    z_QSOs = np.asarray(z_QSOs)

    EW_2796s_cen = np.asarray(EW_2796s_cen); EW_2796s_low = np.asarray(EW_2796s_low); EW_2796s_high = np.asarray(EW_2796s_high)
    EW_2803s_cen = np.asarray(EW_2803s_cen); EW_2803s_low = np.asarray(EW_2803s_low); EW_2803s_high = np.asarray(EW_2803s_high)

    z_ABS_cen = np.asarray(z_ABS_cen); z_ABS_low = np.asarray(z_ABS_low); z_ABS_high = np.asarray(z_ABS_high)
    Amp1s_cen = np.asarray(Amp1s_cen); Amp1s_low = np.asarray(Amp1s_low); Amp1s_high = np.asarray(Amp1s_high)
    Amp2s_cen = np.asarray(Amp2s_cen); Amp2s_low = np.asarray(Amp2s_low); Amp2s_high = np.asarray(Amp2s_high)
    StdDev1s_cen = np.asarray(StdDev1s_cen); StdDev1s_low = np.asarray(StdDev1s_low); StdDev1s_high = np.asarray(StdDev1s_high)
    StdDev2s_cen = np.asarray(StdDev2s_cen); StdDev2s_low = np.asarray(StdDev2s_low); StdDev2s_high = np.asarray(StdDev2s_high)

    LineSNRmins = np.asarray(LineSNRmins)
    LineSNRmaxs = np.asarray(LineSNRmaxs)
    ContinuumMethods = np.asarray(ContinuumMethods)


    # prepare to write out fits file
    fits = fitsio.FITS(out_fn,'rw')

    # can also be a list of ordinary arrays if you send the names
    array_list = [TARGETIDs,RAs,DECs,Surveys,ZWarns,TSNR_QSOs,TSNR_LYAs,TSNR_LRGs,z_QSOs,EW_2796s_cen,EW_2803s_cen,                    EW_2796s_low,EW_2803s_low,EW_2796s_high,EW_2803s_high,z_ABS_cen,Amp1s_cen,Amp2s_cen,StdDev1s_cen,StdDev2s_cen,                    z_ABS_low,Amp1s_low,Amp2s_low,StdDev1s_low,StdDev2s_low,z_ABS_high,Amp1s_high,Amp2s_high,StdDev1s_high,StdDev2s_high,ContinuumMethods,LineSNRmins,LineSNRmaxs]

    fits.write(array_list, names=names,units=Catalog_units,extname='{}_ABSORBERS'.format(which_line))

    fits.close()

#for testing
#write_hp_catalog(hp_incomplete[0])

#for hp in hp_incomplete:
#    write_hp_catalog(hp)

pool = multiprocessing.Pool(processes = nprocesses)
print('{} processes being utilized'.format(nprocesses))
#run pool
pool.map(write_hp_catalog, hp_incomplete)