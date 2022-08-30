# QSO_AB
QSO-ALS Mg-II pipeline development

The survey validation (SV) phase of DESI will produce a large (~30000) and extremely high S/N and resolution sample of quasar spectra. In this project we will analyze these spectra, via an additional pipeline, for the Mg-II absorption feature from an intervening galaxy. We will then determine extinction curves and relative abundances for these intervening galaxies following the methods of York et al. 2006 (1). We anticipate that ~10% of all SV QSO spectra will show Mg-II absorption, providing a sample of approximately 3000 targets. This analysis will provide an otherwise inaccessible view of galaxies between redshifts 1 and 2. [1] https://ui.adsabs.harvard.edu/abs/2006MNRAS.367..945Y/abstract

Workflow:
-AllSteps_HP files are run used scripts stored in /Scripts. This defaults to running over 10 nodes with each handiling 10% of healpix directories.
-Write_Abs_Catalogs writes individual healpix-based MgII_absorber catalogs. Then indiviudal catalogs are combined into a per-release catalog
-Eval_MgII_Cats generates a series of plots by which to evaluate the catalogs
