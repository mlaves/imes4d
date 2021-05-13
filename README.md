# Automated 3D registration of optical coherence tomography volumes for panoramic imaging

Max-Heinrich Laves, Lüder A. Kahrs, Tobias Ortmaier

If you use this code, please cite:
 
Laves, MH., Kahrs, LA., Ortmaier, T. Volumetric 3D stitching of optical coherence tomography volumes. Current Directions in Biomedical Engineering, 4(1), pp. 327-330. Retrieved 5 Dec. 2018, doi:[10.1515/cdbme-2018-0079](https://doi.org/10.1515/cdbme-2018-0079)

## Abstract

Optical coherence tomography (OCT) is a non-invasive medical imaging modality, which provides high-resolution transectional images of biological tissue. However, its potential is limited due to a relatively small field of view. To overcome this drawback, we describe a scheme for fully automated stitching of multiple 3D-OCT volumes for panoramic imaging.
The voxel displacements between two adjacent images are calculated by extending the Lucas-Kanade optical flow algorithm to dense volumetric images. A RANSAC robust estimator is used to obtain rigid transformations out of the resulting flow vectors. The images are transformed into the same coordinate frame and overlapping areas are blended.
The accuracy of the proposed stitching scheme is evaluated on two datasets of 7 and 4 OCT volumes, respectively. By placing the specimen on a high-accuracy motorized translational stage, ground truth transformations are available. This results in a mean translational error between two adjacent volumes of 16.6±0.8 µm (2.8±0.13 voxel).
To the author’s knowledge, this is the first reported stitching of 3D-OCT volumes by using dense voxel information in the registration process. The achieved results are sufficient for providing high accuracy OCT panoramic images. Combined with a recently available high-speed 4D-OCT, our method enables interactive stitching of free-hand acquired data.

