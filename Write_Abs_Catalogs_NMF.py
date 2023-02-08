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
med_filt_size=99

#implement arg passing to specify if MgII or SiIV catalogs
first_line_wave=2796.3543
second_line_wave=2803.5315
center=(second_line_wave+first_line_wave)/2

ndim=5

#units of columns in resulting catalogs
MgII_cat_units = ['',"deg","deg",'','','','','','','angstrom','angstrom','angstrom','angstrom','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','','']

#names of columns in resulting catalogs
names =['TARGETID','RA','DEC','SURVEY','ZWARN','TSNR2_QSO','TSNR2_LYA','TSNR2_LRG','Z_QSO','EW_2796','EW_2803','EW_2796_ERR_LOW','EW_2803_ERR_LOW','EW_2796_ERR_HIGH','EW_2803_ERR_HIGH','Z_MGII','AMP_2796','AMP_2803','STDDEV_2796','STDDEV_2803','Z_MGII_ERR_LOW','AMP_2796_ERR_LOW','AMP_2803_ERR_LOW','STDDEV_2796_ERR_LOW','STDDEV_2803_ERR_LOW','Z_MGII_ERR_HIGH','AMP_2796_ERR_HIGH','AMP_2803_ERR_HIGH','STDDEV_2796_ERR_HIGH','STDDEV_2803_ERR_HIGH','CONTINUUM_METHOD','LINE_SNR_MIN','LINE_SNR_MAX']

#Example chains_fps
#'/pscratch/sd/l/lucasnap/MgII_Abs_Chains/fuji-NMF'
#'/global/cscratch1/sd/lucasnap/MgII_Abs_Chains'

#file path setup for MgII chains
chains_fp = str(sys.argv[1])

redux = str(sys.argv[2])

#file path setup for reduction directories
reduction_base_dir='/global/cfs/cdirs/desi/spectro/redux/{}/'.format(redux)
summary_cat_dir=os.path.join(reduction_base_dir,'zcatalog')

#reading all survey zcats and forming a dictionary to easily access when needed
if(redux == 'fuji'):
    sv1_zcat=fitsio.read('{}/zpix-{}-dark.fits'.format(summary_cat_dir,'sv1'),'ZCATALOG')
    sv2_zcat=fitsio.read('{}/zpix-{}-dark.fits'.format(summary_cat_dir,'sv2'),'ZCATALOG')
    sv3_zcat=fitsio.read('{}/zpix-{}-dark.fits'.format(summary_cat_dir,'sv3'),'ZCATALOG')
    
    zcat_dic={'sv1':sv1_zcat,'sv2':sv2_zcat,'sv3':sv3_zcat}
    
if(redux == 'guadalupe'):
    main_zcat=fitsio.read('{}/zpix-{}-dark.fits'.format(summary_cat_dir,'main'),'ZCATALOG')
    
    zcat_dic={'main':main_zcat}

#reading QSOcat
QSOcat_fp = '/global/cfs/cdirs/desi/users/edmondc/QSO_catalog/{}/QSO_cat_{}_healpix.fits'.format(redux,redux)
QSOcat = fitsio.read(QSOcat_fp,'QSO_CAT')


#lists to store the healpix directories with/without written .fits MgII_catalogs
hp_complete = []
hp_incomplete = []

for short_hp in os.listdir(chains_fp):
        
        short_fp=os.path.join(chains_fp,short_hp)
        
        
        #check if filepath is a directory
        #if not move to next entry
        if not os.path.isdir(short_fp):
            continue

        for healpix in os.listdir(short_fp):
            
            #output prep, removing file if it exists to avoid appending rather than overwriting
            out_fn='{}/MgII-Absorbers-{}.fits'.format(os.path.join(chains_fp,short_hp,healpix),healpix)
            

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
elif(redux=='guadalupe'):
    possible_srv=['main']

#for healpix in hp_complete:
    
def write_hp_catalog(healpix):

    #healpixel abbreviation
    short_hp=healpix[:-2]

    #output prep, removing file if it exists to avoid appending rather than overwriting
    out_fn='{}/MgII-Absorbers-{}.fits'.format(os.path.join(chains_fp,short_hp,healpix),healpix)

    #go grab spectra file
    coadd_dic={}
    for survey in possible_srv:
        hp_dir=os.path.join(reduction_base_dir,'healpix',survey,'dark',healpix[0:-2],healpix)
        specfile=os.path.join(hp_dir,'coadd-{}-dark-{}.fits'.format(survey,healpix))

        if os.path.exists(specfile):
            specobj = desispec.io.read_spectra(specfile)

            coadd_dic[survey] = specobj


    #Stupid dumb way to hold lists of values
    TARGETIDs=[]
    RAs=[]
    DECs=[]
    Surveys=[]
    ZWarns=[]
    TSNR_QSOs=[]
    TSNR_LYAs=[]
    TSNR_LRGs=[]
    z_QSOs=[]

    EW_2796s_cen=[]; EW_2796s_low=[]; EW_2796s_high=[]
    EW_2803s_cen=[]; EW_2803s_low=[]; EW_2803s_high=[]

    z_MgIIs_cen=[]; z_MgIIs_low=[]; z_MgIIs_high=[]
    Amp1s_cen=[]; Amp1s_low=[]; Amp1s_high=[]
    Amp2s_cen=[]; Amp2s_low=[]; Amp2s_high=[]
    StdDev1s_cen=[]; StdDev1s_low=[]; StdDev1s_high=[]
    StdDev2s_cen=[]; StdDev2s_low=[]; StdDev2s_high=[]

    LineSNRmins=[]
    LineSNRmaxs=[]
    
    ContinuumMethods=[]



    h5_fn='MgII-Abs-Chains-{}.hdf5'.format(healpix)
    h5_fp=os.path.join(chains_fp,short_hp,healpix,h5_fn)
    try:
        h5_file=h5py.File(h5_fp,'r')
    except:
        print('Broken file: {}'.format(h5_fp))
        return

    #print(h5_file.keys())
    for key in h5_file.keys():

        TARGETID,survey,hp,z_MgII,z_QSO,snr1,snr2,contmeth=key.split('_')

        #excluding special survey at the moment
        if(survey=='special'):
            continue

        #grab correct summary catalog and coadd file
        coadd=coadd_dic[survey]
        summary_cat=zcat_dic[survey]
        #find entry in summary catalog
        sum_cat_idx=np.where(summary_cat['TARGETID']==int(TARGETID))[0]
        specobj_idx=np.where(coadd.target_ids() == int(TARGETID))[0]

        #grab correct spectra and coadd (over cameras)
        spec=coadd_cameras(coadd[specobj_idx])

        #grab flux values
        x_spc = spec.wave["brz"]
        y_flx = spec.flux["brz"][0]

        #estimate continuum using median filter
        cont_est = medfilt(y_flx,med_filt_size)

        #find correct index for center of doublet
        index=np.abs(x_spc - (center*(1+float(z_MgII)))).argmin()
        #estimate continuum at this center
        continuum=float(cont_est[index])

        #en["NEW_EW2796"]=np.abs(en['AMP_2796'])*en['STDDEV_2796']*np.sqrt(2*np.pi)/(1.0+en['Z_MGII'])/continuum
        #en["NEW_EW2803"]=np.abs(en['AMP_2803'])*en['STDDEV_2803']*np.sqrt(2*np.pi)/(1.0+en['Z_MGII'])/continuum

        #pull info from summary catalog, ra, dec, rmag
        ra=summary_cat['TARGET_RA'][sum_cat_idx]
        dec=summary_cat['TARGET_DEC'][sum_cat_idx]

        ZWarn=summary_cat['ZWARN'][sum_cat_idx]

        TSNR_QSO=summary_cat['TSNR2_QSO'][sum_cat_idx]
        TSNR_LYA=summary_cat['TSNR2_LYA'][sum_cat_idx]
        TSNR_LRG=summary_cat['TSNR2_LRG'][sum_cat_idx]

        #RFlux=summary_cat['FLUX_R'][sum_cat_idx]

        #grab datasets assocaited with key
        dset = h5_file[key]
        #grab datasets keys
        dkeys=dset.keys()

        #hold attributes of MgII abs. fit
        feature_mid=np.zeros(ndim)
        feature_low=np.zeros(ndim)
        feature_high=np.zeros(ndim)

        for i,dkey in zip(range(ndim),dkeys):
            feature_mid[i]=np.percentile(dset[dkey],50)
            feature_low[i]=np.percentile(dset[dkey],16)
            feature_high[i]=np.percentile(dset[dkey],84)


        #calculate line equivalent widths using amplitudes, widths, redshift and line intercept* (TODO: evaluate this choice?)
        EW_2796s=np.abs(np.random.choice(dset['Amp1'],10000))*np.random.choice(dset['StdDev1'],10000)*np.sqrt(2*np.pi)            /(1+np.random.choice(dset['z'],10000))/continuum

        EW_2803s=np.abs(np.random.choice(dset['Amp2'],10000))*np.random.choice(dset['StdDev2'],10000)*np.sqrt(2*np.pi)            /(1+np.random.choice(dset['z'],10000))/continuum

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
        

        z_MgIIs_cen.append(feature_mid[4]); z_MgIIs_low.append(feature_mid[4]-feature_low[4]); z_MgIIs_high.append(feature_high[4]-feature_mid[4])
        Amp1s_cen.append(feature_mid[0]); Amp1s_low.append(feature_mid[0]-feature_low[0]); Amp1s_high.append(feature_high[0]-feature_mid[0])
        Amp2s_cen.append(feature_mid[1]); Amp2s_low.append(feature_mid[1]-feature_low[1]); Amp2s_high.append(feature_high[1]-feature_mid[1])
        StdDev1s_cen.append(feature_mid[2]); StdDev1s_low.append(feature_mid[2]-feature_low[2]); StdDev1s_high.append(feature_high[2]-feature_mid[2])
        StdDev2s_cen.append(feature_mid[3]); StdDev2s_low.append(feature_mid[3]-feature_low[3]); StdDev2s_high.append(feature_high[3]-feature_mid[3])

        LineSNRmins.append(np.min([float(snr1),float(snr2)]))
        LineSNRmaxs.append(np.max([float(snr1),float(snr2)]))
        
        ContinuumMethods.append(contmeth)

    TARGETIDs=np.asarray(TARGETIDs)
    RAs=np.asarray(RAs)
    DECs=np.asarray(DECs)
    Surveys=np.asarray(Surveys)
    ZWarns=np.asarray(ZWarns)      
    TSNR_QSOs=np.asarray(TSNR_QSOs)
    TSNR_LYAs=np.asarray(TSNR_LYAs)
    TSNR_LRGs=np.asarray(TSNR_LRGs)  
    z_QSOs=np.asarray(z_QSOs)

    EW_2796s_cen=np.asarray(EW_2796s_cen); EW_2796s_low=np.asarray(EW_2796s_low); EW_2796s_high=np.asarray(EW_2796s_high)
    EW_2803s_cen=np.asarray(EW_2803s_cen); EW_2803s_low=np.asarray(EW_2803s_low); EW_2803s_high=np.asarray(EW_2803s_high)

    z_MgIIs_cen=np.asarray(z_MgIIs_cen); z_MgIIs_low=np.asarray(z_MgIIs_low); z_MgIIs_high=np.asarray(z_MgIIs_high)
    Amp1s_cen=np.asarray(Amp1s_cen); Amp1s_low=np.asarray(Amp1s_low); Amp1s_high=np.asarray(Amp1s_high)
    Amp2s_cen=np.asarray(Amp2s_cen); Amp2s_low=np.asarray(Amp2s_low); Amp2s_high=np.asarray(Amp2s_high)
    StdDev1s_cen=np.asarray(StdDev1s_cen); StdDev1s_low=np.asarray(StdDev1s_low); StdDev1s_high=np.asarray(StdDev1s_high)
    StdDev2s_cen=np.asarray(StdDev2s_cen); StdDev2s_low=np.asarray(StdDev2s_low); StdDev2s_high=np.asarray(StdDev2s_high)

    LineSNRmins=np.asarray(LineSNRmins)
    LineSNRmaxs=np.asarray(LineSNRmaxs)
    ContinuumMethods=np.asarray(ContinuumMethods)


    # prepare to write out fits file
    fits = fitsio.FITS(out_fn,'rw')

    # can also be a list of ordinary arrays if you send the names
    array_list=[TARGETIDs,RAs,DECs,Surveys,ZWarns,TSNR_QSOs,TSNR_LYAs,TSNR_LRGs,z_QSOs,EW_2796s_cen,EW_2803s_cen,                    EW_2796s_low,EW_2803s_low,EW_2796s_high,EW_2803s_high,z_MgIIs_cen,Amp1s_cen,Amp2s_cen,StdDev1s_cen,StdDev2s_cen,                    z_MgIIs_low,Amp1s_low,Amp2s_low,StdDev1s_low,StdDev2s_low,z_MgIIs_high,Amp1s_high,Amp2s_high,StdDev1s_high,StdDev2s_high,ContinuumMethods,LineSNRmins,LineSNRmaxs]

    fits.write(array_list, names=names,clobber=True,units=MgII_cat_units,extname='MGII_ABSORBERS')

    fits.close()

pool = multiprocessing.Pool(processes = nprocesses)
print('{} processes being utilized'.format(nprocesses))
#run pool
pool.map(write_hp_catalog, hp_incomplete)