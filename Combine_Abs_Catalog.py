import numpy as np
import os
import fitsio
import sys


#file path setup for MgII chains
chains_fp = str(sys.argv[1])

#'/pscratch/sd/l/lucasnap/MgII_Abs_Chains/fuji-NMF'
#'/global/cscratch1/sd/lucasnap/MgII_Abs_Chains/fuji'

chain_spt = chains_fp.split('/')

redux = chain_spt[-1].split('-')[0]
print('Redux = {}'.format(redux))

#Elliptical Selection Params
#h,k,a,b,A=[1.2,1.2,1,1.2,np.pi/3]

MgII_cat_units = ['',"deg","deg",'','','','','','','angstrom','angstrom','angstrom','angstrom','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','1e-17 erg / (Angstrom cm2 s)','1e-17 erg / (Angstrom cm2 s)','angstrom','angstrom','','','']


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


#Merge individual healpix MgII_catalogs into master catalog
cat_started=False

for healpix in hp_complete:
    
    if int(healpix) < 100:
        short_hp = '0'
    else:
        short_hp = healpix[0:-2]
    
    cat_fp=os.path.join(chains_fp,short_hp,healpix,'MgII-Absorbers-{}.fits'.format(healpix))
    
    #print(cat_fp)

    try:
        MgII_cat = fitsio.read(cat_fp)
    except OSError as e:
        #if empty fits file
        if str(e) == 'No extensions have data':
            continue
        else:
            print(e)
    
    if(cat_started):
        #MgII_cat_full=MgII_cat
        MgII_cat_full=np.concatenate((MgII_cat_full,MgII_cat),dtype=MgII_cat_full.dtype)
    else:
        MgII_cat_full=MgII_cat
        cat_started=True

print('Initial length: {}'.format(len(MgII_cat_full)))
print(MgII_cat_full.dtype.names)

out_fn='/global/cfs/cdirs/desi/users/lucasnap/Temp-Catalogs/MgII-Absorbers-{}-nocuts.fits'.format(chain_spt[-1])
    

#for fuji it is necessary to select the best entry (based on TSNR2_QSO)
if(redux =='fuji'):
    #determine occurance #s for every unique targetID
    TIDs, TID_inv, TID_cnts = np.unique(MgII_cat_full['TARGETID'],return_counts=True,return_inverse=True)
    
    #select those TARGETIDs that only appear once, we don't need to check these
    MgII_oneabs = MgII_cat_full[TID_cnts[TID_inv] == 1]
    
    #start final catalog with these entries
    MgII_final = MgII_oneabs
    
    #iterate over TIDs that appear more than once (should really only be TIDs that appear more than once and with different surveys
    for TID in TIDs[TID_cnts > 1]:
        #select indices from full catalog with this targetid
        inds = np.where(MgII_cat_full['TARGETID'] == TID)[0]
        #select surveys
        srv = MgII_cat_full['SURVEY'][inds]

        #if only one survey we can just add those
        if len(np.unique(srv))==1:
            #just concatenate to oneabs
            MgII_final = np.concatenate((MgII_final,MgII_cat_full[inds]),dtype=MgII_final.dtype)
        else:
            #determined best TSNR2 value
            TSNR2_best = np.max(MgII_cat_full[inds]['TSNR2_QSO'])
            #build mask to select only entries with this value
            Is_best = MgII_cat_full[inds]['TSNR2_QSO']==TSNR2_best
            #append those entries
            MgII_final = np.concatenate((MgII_final,MgII_cat_full[inds][Is_best]),dtype=MgII_final.dtype)

    print(len(MgII_final))
    
    if os.path.exists(out_fn):
        print('Removing old no-cuts catalog')
        os.remove(out_fn)
        
    fitsio.write(out_fn,MgII_final,extname='MGII_ABSORBERS',units=MgII_cat_units)
    
elif len(np.unique(MgII_cat_full['SURVEY'])) == 1:
    if os.path.exists(out_fn):
        print('Removing old no-cuts catalog')
        os.remove(out_fn)
        
    fitsio.write(out_fn,MgII_cat_full,extname='MGII_ABSORBERS',units=MgII_cat_units)