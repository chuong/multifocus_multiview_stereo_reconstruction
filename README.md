# multifocus_multiview_stereo_reconstruction
Data and codes for multifocus multiview stereo reconstruction

Please cite the following paper if you use the codes and data:

Chowdhury, S.A.H., Nguyen, C., Li, H. et al. Fixed-Lens camera setup and calibrated image registration for multifocus multiview 3D reconstruction. Neural Comput & Applic (2021). https://doi.org/10.1007/s00521-021-05926-7

Data:

Moving lens data and image acquisition setup can be found in the following links.

Real: https://cloudstor.aarnet.edu.au/plus/s/NCeCigvr7bqJlEa

Synthetic: https://cloudstor.aarnet.edu.au/plus/s/LDgHE3Zkex64DD6

Fixed lens data and image acquisition setup can be found in the following links.

Real: https://cloudstor.aarnet.edu.au/plus/s/WrDAPtfiKi6DbTa

Synthetic: https://cloudstor.aarnet.edu.au/plus/s/IhJcUT2hPLG7ja5

Data associated to synthetic data generation using Blender can be found in the following link.

https://cloudstor.aarnet.edu.au/plus/s/6zAdDX0BwDuSOLH

Workflow:

1. Computing homography transformation via clibration

run calibration_moving_lens_shiftscale_real.py for obtaining homography transformation to align the real moving lens multifocus images

run calibration_moving_lens_shiftscale_forblender.py for obtaining homography transformation to align the synthetic moving lens multifocus images

run calibration_fixed_lens_shiftscale_real.py for obtaining homography transformation to align the real fixed lens multifocus images

run calibration_fixed_lens_shiftscale_forblender.py for obtaining homography transformation to align the synthetic fixed lens multifocus images

run calibration_fixed_lens_Li_2019_forblender.py for obtaining fitted homography transformation proposed by Li and Nguyen (2019) for fixed lens.

2. Perform multificus image stacking

run run.py for multifocus stacking with the appropriate homography transfromation function

If homography transformation is not privided for image alignment, select feature based alignment based on SIFT feature.

This code also supports ECC algorithm for image alignment.



Please select pyramid fusion to apply Laplacian pyramid fusion method.

This code also supports additional fusion methods including guided filtering approach.

4. Run Background_Subtraction.py to obtain transparent background of the stacked in-focus images.

5. Perform 3D reconstruction using the stacked images by a 3D reconstruction software namely Agisoft of Photoscan or Meshroom of Alicevision.

6. Then perform quantitative analysis with camera poses provided by the 3D reconstruction software.

run positions.py for resonstruction from real images.

run positions_blender_controlled_camera_pose.py for reconstruction from synthetic images with complete spherical camera distribution.

run positions_blender_controlled_camera_pose_incomplete_sphere.py for reconstruction from synthetic images with incomplete spherical camera distribution.

References:
Li, Hengjia and Nguyen, Chuong, 2019. Perspective-consistent multifocus multiview 3D reconstruction of small objects. In 2019 Digital Image Computing: Techniques and ApplicatLi, Hengjia and Nguyen, Chuong, 2019. Perspective-consistent multifocus multiview 3D reconstruction of small objects. In 2019 Digital Image Computing: Techniques and Applications (DICTA), pages 1–8. IEEE.
