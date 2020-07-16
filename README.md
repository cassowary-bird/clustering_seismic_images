Seismic images have different quality. This quality approximately means is seismic image good or bad in terms of appropriate geological horizons.

Some approach for seismic images clustering according to the quality is introduced here. 

Features are some signal parameters of seismic data and characteristics based on computer vision algorithms. 

Clusterization is performed by OPTICS algorithm with the previous combination of dimensionality reduction techniques (PCA, Isomap, Autoencoder).

All methods and import necessary libraries are located in 'all_methods.ipynb' file. It is assumed that there is file with seismic amplitudes (.npy file) on google disk. A bit more information about order of using algorithms provides in 'clustering_methods_(verbose).ipynb' file. Applying above methods to seismic data are located in files 'clustering_with_no_reduction_for_features1.ipynb', 'clustering_with_PCA_for_features1.ipynb', 'clustering_with_autoencoder_for_features1.ipynb', 'clustering_with_Isomap_for_features1.ipynb', 'clustering_with_t-sne_or_lle_for_features1.ipynb', 'clustering_for_features2.ipynb'.
