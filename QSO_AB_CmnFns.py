#DESIGNED TO BE "IMPORTED" INTO JUPYTER NBS via:
# %run /global/u2/l/lucasnap/git/QSO_AB/QSO_AB_CmnFns.py

import os
import numpy as np
import matplotlib.pyplot as plt
import desispec.io

from desispec.coaddition import coadd_cameras
from desimodel.footprint import radec2pix
from scipy.signal import medfilt

from IPython.display import clear_output

plt.rcParams.update({'font.size': 24})

first_line_wave=2796.3543
second_line_wave=2803.5315

def get_spec_healpix(targetid,healpix,survey,redux,coadded=True):
    
    reduction_base_dir = '/dvs_ro/cfs/cdirs/desi/spectro/redux/{}/'.format(redux)
    healpix = str(healpix)
    
    #form full file paths
    hp_dir = os.path.join(reduction_base_dir,'healpix',survey,'dark',healpix[0:-2],healpix)
    specfile = os.path.join(hp_dir,'coadd-{}-dark-{}.fits'.format(survey,healpix))

    if(os.path.exists(specfile)):
        
        #read fibermap and find correct index based on targetid
        specobj = desispec.io.read_spectra(specfile)
        specobj_idx = np.where(specobj.target_ids() == targetid)
        
        if coadded:
            return(coadd_cameras(specobj[specobj_idx]))
        else:
            return(specobj[specobj_idx])
    
    else:
        return
    
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
def smooth_spectra(y_flx,smooth_strength):
    #set up kernel using given strength
    kernel = Gaussian1DKernel(stddev=smooth_strength)
    #smooth spectra
    y_smooth=convolve(y_flx,kernel)
    
    return(y_smooth)

#Elliptical Selection Params
#h,k,a,b,A=[1.2,1.2,1,1.2,np.pi/3]

#trying new selection params
h,k,a,b,A=[1.4,1.4,0.9,0.9,np.pi/3]

#Purity cuts developed from visual inspection. Accepts MgII_catalog and returns MgII_catalog following cuts
def purity_cuts(MgII_cat,pre_ellip_cut=False,PIA_cut=True,verbose=True):
    
    if PIA_cut:
        if 'Z_MGII' in MgII_cat.dtype.names:
            zdop = (MgII_cat['Z_MGII']-MgII_cat['Z_QSO'])/(1+MgII_cat['Z_QSO'])
        elif 'Z_ABS' in MgII_cat.dtype.names:
            zdop = (MgII_cat['Z_ABS']-MgII_cat['Z_QSO'])/(1+MgII_cat['Z_QSO'])
        if verbose:
            print('Removing {} physically impossible absorption systems'.format(np.sum((zdop > (5000/300000)))))
        MgII_cat = MgII_cat[zdop < (5000/300000)]
    
    int_size=len(MgII_cat)
    
    #first cut positive amplitudes
    if np.mean(MgII_cat['AMP_2796'] < 0):
        MgII_cat =  MgII_cat[MgII_cat['AMP_2796']<0]
        MgII_cat = MgII_cat[MgII_cat['AMP_2803']<0]
    elif np.mean(MgII_cat['AMP_2796'] > 0):
        MgII_cat =  MgII_cat[MgII_cat['AMP_2796']>0]
        MgII_cat = MgII_cat[MgII_cat['AMP_2803']>0]
    
    #Print results and save new catalog size
    if verbose:
        print('Result of Positive Amplitude Cuts: {} systems removed'.format(int_size-len(MgII_cat)))
    sec_size = len(MgII_cat)

    
    #Print results and save new catalog size
    #print('Result of Large Negative Spike Cuts: {} systems removed'.format(sec_size-len(MgII_cat)))
    ter_size = len(MgII_cat)
    
    #Finally perform elliptical selection
    xt = MgII_cat['AMP_2796']/MgII_cat['AMP_2803']
    yt = MgII_cat['STDDEV_2796']/MgII_cat['STDDEV_2803']

    #gives number of systems post initial cuts that lie outside selection
    ellipse_mask = ((xt-h)*np.cos(A)+(yt-k)*np.sin(A))**2/a**2+((xt-h)*np.sin(A)-(yt-k)*np.cos(A))**2/b**2>1

    #resulting #systems given by 
    if verbose:
        print('Result of Elliptical Cuts: {} systems removed'.format(np.sum(ellipse_mask)))
    
    #return catalog results before elliptical cuts for purposes of plotting and actual number of systems post-cut for numbers
    if(pre_ellip_cut):
        return(MgII_cat,int_size-(int_size-ter_size+np.sum(ellipse_mask)))
    else:
        print('Returning catalog with {} entries'.format(MgII_cat[~ellipse_mask].size))
        return(MgII_cat[~ellipse_mask])

def MgII_Model(theta,x):
    z,a1,a2,s1,s2 = theta
    #determine peak centers
    m1 = (z+1)*first_line_wave
    m2 = (z+1)*second_line_wave
    
    #Generate Model
    model = a1*np.exp((-(x-m1)**2)/(2*s1**2))+a2*np.exp((-(x-m2)**2)/(2*s2**2))
    return model

def plot_abs_spectra(spectra,z_abs=0,title=''):
    x_spc=spectra.wave['brz']
    y_flx=smooth_spectra(spectra.flux['brz'][0],2)
    y_err=1/np.sqrt(spectra.ivar['brz'][0])

    plt.figure(figsize=(20,5))
    plt.plot(x_spc,y_flx,alpha=0.8)
    plt.plot(x_spc,y_err,alpha=0.8)
    plt.title(title)

    plt.xlim(x_spc[0],x_spc[-1])
    
    bot=np.min([0.0,np.percentile(y_flx,0.1)])
    plt.ylim(bot,np.percentile(y_flx,99.9))
    
    plt.ylabel('Flux')
    plt.xlabel('Wavelength (Å)')
    
    if(len(z_abs)>0):
        for z in z_abs:
            plt.fill_betweenx([-100,100], first_line_wave*(1+float(z))-50, x2=second_line_wave*(1+float(z))+30, alpha=0.2, color='gray')
    else:
        plt.fill_betweenx([-100,100], first_line_wave*(1+float(z_abs))-50, x2=second_line_wave*(1+float(z_abs))+30, alpha=0.2, color='gray')

    plt.show()