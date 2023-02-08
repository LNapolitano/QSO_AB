# QSO_AB
QSO-ALS Mg-II pipeline development

The survey validation (SV) phase of DESI will produce a large (~30000) and extremely high S/N and resolution sample of quasar spectra. In this project we will analyze these spectra, via an additional pipeline, for the Mg-II absorption feature from an intervening galaxy. We will then determine extinction curves and relative abundances for these intervening galaxies following the methods of York et al. 2006 (1). We anticipate that ~10% of all SV QSO spectra will show Mg-II absorption, providing a sample of approximately 3000 targets. This analysis will provide an otherwise inaccessible view of galaxies between redshifts 1 and 2. [1] https://ui.adsabs.harvard.edu/abs/2006MNRAS.367..945Y/abstract

---------Release Notes---------
1.0: Adjusted line fitting technique, lines are now fit using a QSO continuum estimated using https://github.com/guangtunbenzhu/NonnegMFPy NMF code. As a fallback when a good continuum (chi2 > 4.0) cannot be found using the NMF technique we use a wide (85 pixel) median filter. As a result the fit is reduced to 5 parameters from 7. Input QSO catalog has been adjusted to only include QSO targets in interest of higher purity. Additionally adjusted line_snr detection thresholds in effort to improve purity. Previously required candidate doublet lines to have snrs > 3 and > 1.5 respectively. By lowering the first value to > 2.5 we improve catalog completeness by nearly 10% with essentially no change to purity
-------------------------------


Workflow:

-MgII_NMF_HP.py writes .hdf5 files for each healpixel within a given survey, these contain the MgII_absorber chains as well as info regarding the spectrum.

-Write_Abs_Catalogs_NMF.py writes .fits files for each healpixel from the .hdf5 files

-Combine_Abs_Catalog.py combines these .fits files into one file, resolving overlapping entries

-Eval_MgII_Cats produces some basic plots as well as applying the purity cuts detailed in QSO_AB_CmnFns.py