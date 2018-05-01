"""Experiment 7.
Run LAP modified with frenet from single example outside BL. No previous alignment.
"""

from __future__ import print_function
import nibabel as nib
import numpy as np
from nibabel.streamlines import load, save
from tractograms_slr import tractograms_slr
from dipy.tracking.streamline import apply_affine
from lap_single_example import compute_kdtree_and_dr_tractogram, save_bundle
from dipy.tracking.distances import bundles_distances_mam
from compute_voxel_measures import compute_voxel_measures
from test_rlap_frenet import compute_lap_inputs, RLAP_modified
from dipy.tracking.streamline import length, set_number_of_points
from test_shape import frenet_diff


if __name__ == '__main__':

    experiment = 'exp7' #'test' #'exp1'
    sub_list = ['990366', '991267', '992673', '992774', '993675']#, '995174', '996782']
    tract_name_list = ['Left_IFOF', 'Left_ILF', 'Left_Arcuate', 'Callosum_Forceps_Minor'] # 'Right_Cingulum_Cingulate', 'Callosum_Forceps_Major']
    example_list = ['615441', '615744', '616645', '617748', '618952'] 
    src_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment
    distance_func = bundles_distances_mam
    h_list = [1, 0.8, 0.6, 0.4, 0.2, 0]
    k = 500

    np.random.seed(0)

    DSC_rlap_frenet = np.zeros((len(sub_list), len(tract_name_list), len(example_list), len(h_list)))
    wDSC_rlap_frenet = np.zeros((len(sub_list), len(tract_name_list), len(example_list), len(h_list)))
    J_rlap_frenet = np.zeros((len(sub_list), len(tract_name_list), len(example_list), len(h_list)))
    sensitivity_rlap_frenet = np.zeros((len(sub_list), len(tract_name_list), len(example_list), len(h_list)))

    for s, sub in enumerate(sub_list):
	
	static_tractogram_filename = '%s/%s/%s_output_fe.trk' %(src_dir, sub, sub)
	static_tractogram = nib.streamlines.load(static_tractogram_filename)
	static_tractogram = static_tractogram.streamlines
	print("Compute the dissimilarity representation of the static tractogram of subject %s and build the kd-tree." %sub)
	kdt, prototypes = compute_kdtree_and_dr_tractogram(static_tractogram, distance=distance_func)

    	for t, tract_name in enumerate(tract_name_list):

	    true_tract_filename = '%s/%s/%s_%s_tract.trk' %(src_dir, sub, sub, tract_name)
	    true_tract = nib.streamlines.load(true_tract_filename)
	    tck_filename = '%s/%s/%s_%s_tract.tck' %(results_dir, sub, sub, tract_name)
	    nib.streamlines.save(true_tract.tractogram, tck_filename)
	    true_tract = true_tract.streamlines

    	    for e, example in enumerate(example_list):
	
		moving_example_filename = '%s/%s/%s_%s_tract.trk' %(src_dir, example, example, tract_name)	
		moving_example = nib.streamlines.load(moving_example_filename)
		moving_example = moving_example.streamlines
		moving_tractogram_filename = '%s/%s/%s_output_fe.trk' %(src_dir, example, example)

		print("Computing the affine slr transformation.")
		affine = tractograms_slr(moving_tractogram_filename, static_tractogram_filename)

		print("Applying the affine to the example bundle.")
		moving_example_aligned = np.array([apply_affine(affine, s) for s in moving_example])

		print("Compute the dissimilarity of the aligned example bundle with the prototypes of static tractogram.")
		dm_moving_example_aligned = distance_func(moving_example_aligned, prototypes)

		print("Segmentation as MODIFIED WITH FRENET Rectangular linear Assignment Problem (RLAP).")
		distance_matrix, frenet_matrix, superset = compute_lap_inputs(kdt, k, dm_moving_example_aligned, moving_example_aligned, static_tractogram, distance_func)
		d_matrix_name = '%s/%s/%s_D_%s_tract_E%s.npy' %(results_dir, sub, sub, tract_name, example)
		f_matrix_name = '%s/%s/%s_F_%s_tract_E%s.npy' %(results_dir, sub, sub, tract_name, example)
		superset_name = '%s/%s/%s_superset_%s_tract_E%s.npy' %(results_dir, sub, sub, tract_name, example)
		np.save(d_matrix_name, distance_matrix)
		np.save(f_matrix_name, frenet_matrix)
		np.save(superset_name, superset)
									
		#normalize matrices
		#frenet_matrix1=(frenet_matrix-np.min(frenet_matrix))/(np.max(frenet_matrix)-np.min(frenet_matrix))
		#distance_matrix1=(distance_matrix-np.min(distance_matrix))/(np.max(distance_matrix)-np.min(distance_matrix))
		frenet_matrix1=(frenet_matrix-np.mean(frenet_matrix))/np.std(frenet_matrix)				
		distance_matrix1=(distance_matrix-np.mean(distance_matrix))/np.std(distance_matrix)

		for hh, h in enumerate(h_list):
			estimated_bundle_idx, min_cost_values = RLAP_modified(distance_matrix1, frenet_matrix1, superset, h)
			estimated_bundle = static_tractogram[estimated_bundle_idx]
			out_filename = '%s/%s/%s_%s_h%s_tract_E%s.tck' %(results_dir, sub, sub, tract_name, int(h*100), example)
			save_bundle(estimated_bundle_idx, static_tractogram_filename, out_filename)
			print("Computing voxel measures.")
			DSC, wDSC, J, sensitivity = compute_voxel_measures(estimated_bundle, true_tract)	
			print("The DSC value with h=%s is %s" %(h,DSC))
			print("The weighted DSC value is %s" %wDSC)
	    		print("The Jaccard index is %s" %J)
	    		print("The sensitivity is %s" %sensitivity)
			
			DSC_rlap_frenet[s, t, e, hh] = DSC
    			wDSC_rlap_frenet[s, t, e, hh] = wDSC
    			J_rlap_frenet[s, t, e, hh] = J
    			sensitivity_rlap_frenet[s, t, e, hh] = sensitivity
			np.save('DSC_rlap_frenet_25', DSC_rlap_frenet)
			np.save('wDSC_rlap_frenet_25', wDSC_rlap_frenet)
			np.save('J_rlap_frenet_25', J_rlap_frenet)
			np.save('sensitivity_rlap_frenet_25', sensitivity_rlap_frenet)
		
