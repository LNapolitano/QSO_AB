import numpy as np
import os
import fitsio
import sys


#file path setup for MgII chains
chains_fp = str(sys.argv[1])

#'/pscratch/sd/l/lucasnap/MgII_Abs_Chains/fuji-NMF'
#'/global/cscratch1/sd/lucasnap/MgII_Abs_Chains/fuji'

redux = str(sys.argv[2])

#Elliptical Selection Params
h,k,a,b,A=[1.2,1.2,1,1.2,np.pi/3]

#Purity cuts developed from visual inspection. Accepts MgII_catalog and returns MgII_catalog following cuts
def purity_cuts(MgII_cat,pre_ellip_cut=False,PIA_cut=True):
    
    if(PIA_cut):
        print('Removing {} physically impossible absorption systems'.format(np.sum((MgII_cat['Z_MGII']>0.05+MgII_cat['Z_QSO']))))
        MgII_cat=MgII_cat[~(MgII_cat['Z_MGII']>0.05+MgII_cat['Z_QSO'])]
    #record inital size for later comparison
    int_size=len(MgII_cat)
    
    #first cut positive amplitudes
    MgII_cat=MgII_cat[MgII_cat['AMP_2796']<0]
    MgII_cat=MgII_cat[MgII_cat['AMP_2803']<0]
    
    #Print results and save new catalog size
    print('Result of Positive Amplitude Cuts: {} systems removed'.format(int_size-len(MgII_cat)))
    sec_size=len(MgII_cat)

    
    #Print results and save new catalog size
    #print('Result of Large Negative Spike Cuts: {} systems removed'.format(sec_size-len(MgII_cat)))
    ter_size=len(MgII_cat)
    
    #Finally perform elliptical selection
    xt = MgII_cat['AMP_2796']/MgII_cat['AMP_2803']
    yt = MgII_cat['STDDEV_2796']/MgII_cat['STDDEV_2803']

    #gives number of systems post initial cuts that lie outside selection
    ellipse_mask=((xt-h)*np.cos(A)+(yt-k)*np.sin(A))**2/a**2+((xt-h)*np.sin(A)-(yt-k)*np.cos(A))**2/b**2>1

    #resulting #systems given by 
    print('Result of Elliptical Cuts: {} systems removed'.format(np.sum(ellipse_mask)))
    print('Result of Purity Cuts: {} systems removed of initial {}\n'.format(int_size-ter_size+np.sum(ellipse_mask),int_size))
    
    #return catalog results before elliptical cuts for purposes of plotting and actual number of systems post-cut for numbers
    if(pre_ellip_cut):
        return(MgII_cat,int_size-(int_size-ter_size+np.sum(ellipse_mask)))
    else:
        return(MgII_cat[~ellipse_mask])


#lists to store the healpix directories with/without written .fits MgII_catalogs
hp_complete=[]
hp_incomplete=[]

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


#Merge individual healpix MgII_catalogs into master catalog
cat_started=False

for healpix in hp_complete:
    short_hp=healpix[:-2]
    
    cat_fp=os.path.join(chains_fp,short_hp,healpix,'MgII-Absorbers-{}.fits'.format(healpix))
    
    #print(cat_fp)

    try:
        MgII_cat=fitsio.read(cat_fp)
    except:# OSError as e:
        #print(e)
        continue
    
    if(cat_started):
        #MgII_cat_full=MgII_cat
        MgII_cat_full=np.concatenate((MgII_cat_full,MgII_cat),dtype=MgII_cat_full.dtype)
    else:
        MgII_cat_full=MgII_cat
        cat_started=True

                
print(len(MgII_cat_full))

#read fujilupe catalog
fujilupe=fitsio.read("/global/cfs/cdirs/desi/science/gqp/zcatalog_summary/zall-pix-fujilupe.fits")

#mask for only the primary observations, based on selection detailed in Fujilupe_CombineZcat.ipynb (location given at top of this notebook section)
is_primary = fujilupe["ZCAT_PRIMARY"]

print('Initial length: {}'.format(len(MgII_cat_full)))
print(MgII_cat_full.dtype.names)

MgII_pure = MgII_cat_full
out_fn='/global/cfs/cdirs/desi/users/lucasnap/MgII-Absorbers-{}-nocuts.fits'.format(redux)
    

#for fuji it is necessary to select the best entry (based on TSNR2_LRG)
if(redux =='fuji'):
    sv1_mask=MgII_pure['SURVEY']=='sv1'
    sv2_mask=MgII_pure['SURVEY']=='sv2'
    sv3_mask=MgII_pure['SURVEY']=='sv3'


    sv1_repeat_TID,sv1_repeat_idx=np.intersect1d(MgII_pure[sv1_mask]['TARGETID'],np.concatenate((MgII_pure[sv2_mask]['TARGETID'],\
                                                                                                 MgII_pure[sv3_mask]['TARGETID'])),return_indices=True)[:2]

    sv2_repeat_TID,sv2_repeat_idx=np.intersect1d(MgII_pure[sv2_mask]['TARGETID'],np.concatenate((MgII_pure[sv1_mask]['TARGETID'],\
                                                                                                 MgII_pure[sv3_mask]['TARGETID'])),return_indices=True)[:2]

    sv3_repeat_TID,sv3_repeat_idx=np.intersect1d(MgII_pure[sv3_mask]['TARGETID'],np.concatenate((MgII_pure[sv1_mask]['TARGETID'],\
                                                                                                 MgII_pure[sv2_mask]['TARGETID'])),return_indices=True)[:2]

    all_repeat_TID=np.unique(np.concatenate((sv1_repeat_TID,sv2_repeat_TID,sv3_repeat_TID)))

    sv1_only_mask=np.ones(len(MgII_pure[sv1_mask]),bool)
    sv1_only_mask[sv1_repeat_idx] = 0
    sv1_only=MgII_pure[sv1_mask][sv1_only_mask]

    sv2_only_mask=np.ones(len(MgII_pure[sv2_mask]),bool)
    sv2_only_mask[sv2_repeat_idx] = 0
    sv2_only=MgII_pure[sv2_mask][sv2_only_mask]

    sv3_only_mask=np.ones(len(MgII_pure[sv3_mask]),bool)
    sv3_only_mask[sv3_repeat_idx] = 0
    sv3_only=MgII_pure[sv3_mask][sv3_only_mask]

    #First find the indices in the fujilupe catalog of the repeatping targetids
    fujilupe_primary_repeat_idx=np.intersect1d(fujilupe[is_primary]['TARGETID'],all_repeat_TID,return_indices=True)[1]
    #next take a subset of fujilupe corresponding tp those indices
    fujilupe_primary_repeat=fujilupe[is_primary][fujilupe_primary_repeat_idx]

    #Now lets determine which survey provides the best observations for this subset
    sv1_best_mask=fujilupe_primary_repeat['SURVEY']=='sv1'
    sv2_best_mask=fujilupe_primary_repeat['SURVEY']=='sv2'
    sv3_best_mask=fujilupe_primary_repeat['SURVEY']=='sv3'

    #Lets grab the subsets of fujilupe for which a certain targetid is best
    sv1_best=fujilupe_primary_repeat[sv1_best_mask]
    sv2_best=fujilupe_primary_repeat[sv2_best_mask]
    sv3_best=fujilupe_primary_repeat[sv3_best_mask]

    sv1_best_TID_idx=np.intersect1d(MgII_pure[sv1_mask]['TARGETID'],sv1_best['TARGETID'],return_indices=True)[1]
    sv2_best_TID_idx=np.intersect1d(MgII_pure[sv2_mask]['TARGETID'],sv2_best['TARGETID'],return_indices=True)[1]
    sv3_best_TID_idx=np.intersect1d(MgII_pure[sv3_mask]['TARGETID'],sv3_best['TARGETID'],return_indices=True)[1]

    sv1_best=MgII_pure[sv1_mask][sv1_best_TID_idx]
    sv2_best=MgII_pure[sv2_mask][sv2_best_TID_idx]
    sv3_best=MgII_pure[sv3_mask][sv3_best_TID_idx]

    print(len(sv1_only),len(sv2_only),len(sv3_only),len(sv1_best),len(sv2_best),len(sv3_best))

    combined_pure_MgIIcat=np.concatenate([sv1_only,sv2_only,sv3_only,sv1_best,sv2_best,sv3_best],dtype=sv1_only.dtype)
    print(len(combined_pure_MgIIcat))
    fitsio.write(out_fn,combined_pure_MgIIcat,extname='MGII_ABSORBERS')

if(redux == 'guadalupe'):
    fitsio.write(out_fn,MgII_pure,extname='MGII_ABSORBERS',units='MgII_cat_units')