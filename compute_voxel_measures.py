"""Compute DSC"""

import nibabel as nib
from nibabel.streamlines import load 
from dipy.tracking.utils import affine_for_trackvis
from dipy.tracking.vox2track import streamline_mapping


def compute_voxel_measures(estimated_tract_filename, true_tract_filename):

	#Loading tracts
	estimated_tract = nib.streamlines.load(estimated_tract_filename)
	true_tract = nib.streamlines.load(true_tract_filename)
	affine_est = estimated_tract.affine
	affine_true = true_tract.affine
	estimated_tract = estimated_tract.streamlines
	true_tract = true_tract.streamlines

	voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=affine_est).keys()
	voxel_list_true_tract = streamline_mapping(true_tract, affine=affine_true).keys()
	TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
	vol_A = len(set(voxel_list_estimated_tract))
	vol_B = len(set(voxel_list_true_tract))
	DSC = 2.0 * float(TP) / float(vol_A + vol_B)

	return DSC, TP, vol_A, vol_B


if __name__ == '__main__':

	#estimated_target_tract_filename = '../initial_tests/0007_Left_Cingulum_Cingulate_tract.trk'
	estimated_target_tract_filename = '../initial_tests/0007_cbL_tract_IU5.trk'
	true_target_tract_filename = '../initial_tests/0007_Left_Cingulum_Cingulate_tract_true.trk'
	
	DSC, TP, vol_A, vol_B = compute_voxel_measures(estimated_target_tract_filename, true_target_tract_filename)

	print("The DSC value is %s" %DSC)

