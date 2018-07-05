"""Estimate a tract with lap-mulitple-examples knowning its size."""

from __future__ import print_function
import nibabel as nib
import numpy as np
import dipy
import os
from dipy.tracking.utils import length
from dipy.tracking.streamline import set_number_of_points
from compute_streamline_measures import compute_roc_curve_lap, compute_y_vectors_lap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def ranking_schema(superset_estimated_target_tract_idx, superset_estimated_target_tract_cost):
    """ Rank all the extracted streamlines estimated by the LAP with multiple examples   
    according to the number of times that they were selected and the total cost. 
    """
    idxs = np.unique(superset_estimated_target_tract_idx)
    how_many_times_selected = np.array([(superset_estimated_target_tract_idx == idx).sum() for idx in idxs])
    how_much_cost = np.array([((superset_estimated_target_tract_idx == idx)*superset_estimated_target_tract_cost).sum() for idx in idxs])
    ranking = np.argsort(how_many_times_selected)[::-1]
    tmp = np.unique(how_many_times_selected)[::-1]
    for i in tmp:
        tmp1 = (how_many_times_selected == i)
        tmp2 = np.where(tmp1)[0]
        if tmp2.size > 1:
            tmp3 = np.argsort(how_much_cost[tmp2])
            ranking[how_many_times_selected[ranking]==i] = tmp2[tmp3]
 
    return idxs[ranking]


def resample_tractogram(tractogram, step_size):
    """Resample the tractogram with the given step size.
    """
    lengths=list(length(tractogram))
    tractogram_res = []
    for i, f in enumerate(tractogram):
	nb_res_points = np.int(np.floor(lengths[i]/step_size))
	tmp = set_number_of_points(f, nb_res_points)
	tractogram_res.append(tmp)
    return tractogram_res


def save_tract(estimated_bundle_idx, static_tractogram_Tractogram, static_tractogram, out_filename):

	extension = os.path.splitext(out_filename)[1]
	aff_vox_to_ras = static_tractogram_Tractogram.affine
	voxel_sizes = static_tractogram_Tractogram.header['voxel_sizes']
	dimensions = static_tractogram_Tractogram.header['dimensions']
	estimated_bundle = static_tractogram[estimated_bundle_idx]
	
	if extension == '.trk':
		print("Saving bundle in %s" % out_filename)
		
		# Creating header
		hdr = nib.streamlines.trk.TrkFile.create_empty_header()
		hdr['voxel_sizes'] = voxel_sizes
		hdr['voxel_order'] = 'LAS'
		hdr['dimensions'] = dimensions
		hdr['voxel_to_rasmm'] = aff_vox_to_ras 

		# Saving bundle
		t = nib.streamlines.tractogram.Tractogram(estimated_bundle, affine_to_rasmm=np.eye(4))
		nib.streamlines.save(t, out_filename, header=hdr)

	elif extension == '.tck':
		print("Saving bundle in %s" % out_filename)

		# Creating header
		hdr = nib.streamlines.tck.TckFile.create_empty_header()
		hdr['voxel_sizes'] = voxel_sizes
		hdr['dimensions'] = dimensions
		hdr['voxel_to_rasmm'] = aff_vox_to_ras

		# Saving bundle
		t = nib.streamlines.tractogram.Tractogram(estimated_bundle, affine_to_rasmm=np.eye(4))
		nib.streamlines.save(t, out_filename, header=hdr)

	else:
		print("%s format not supported." % extension)	



if __name__ == '__main__':

    experiment = 'exp12'
    sub_list = ['910443']#['910241']#['991267']#['983773',#['910241', '910443', '911849']
    #tract_name_list = ['Left_Corticospinal', 'Right_Corticospinal', 'Left_IFOF', 'Right_IFOF', 'Left_Thalamic_Radiation', 'Right_Thalamic_Radiation', 'Left_Arcuate', 'Right_Arcuate']
    tract_name_list = ['Left_pArc', 'Right_pArc', 'Left_TPC', 'Right_TPC', 'Left_MdLF-SPL', 'Right_MdLF-SPL', 'Left_MdLF-Ang', 'Right_MdLF-Ang']
    #true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk_ens_prob_afq'
    true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk_ens_prob_wma'
    #examples_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment
    mode_list = ['orig', 'fake']


    for s, sub in enumerate(sub_list):

	print("Loading tractogram...")
	static_tractogram = '%s/input_result_lap/%s_track.trk' %(results_dir, sub)
	static_tractogram_Tractogram = nib.streamlines.load(static_tractogram)
	static_tractogram = static_tractogram_Tractogram.streamlines
	#print("Resampling tractogram with step size = 0.625 mm")
	#static_tractogram_res = resample_tractogram(static_tractogram, step_size=0.625)
	#static_tractogram = np.array(static_tractogram_res, dtype=np.object)

    	for t, tract_name in enumerate(tract_name_list):
	
  	    true_tract_filename = '%s/%s/%s_%s_tract.trk' %(true_tracts_dir, sub, sub, tract_name)
	    true_tract = nib.streamlines.load(true_tract_filename)
	    true_tract = true_tract.streamlines
	    len_target = len(true_tract)
	
	    #result_lap_filename = '%s/input_result_lap/%s_%s_result_lap_afq5_res.npy' %(results_dir, sub, tract_name)
	    result_lap_filename = '%s/input_result_lap/%s_%s_result_lap_wma5_res.npy' %(results_dir, sub, tract_name)
	    result_lap = np.load(result_lap_filename)
   	    result_lap = np.array(result_lap)
	    estimated_bundle_idx = np.hstack(result_lap[:,0,0])
	    min_cost_values = np.hstack(result_lap[:,0,1])
	    example_bundle_len_med = np.median(np.hstack(result_lap[:,0,2]))
	    estimated_bundle_idx_ranked = ranking_schema(estimated_bundle_idx, min_cost_values)

	    print("Estimating %s with original LAP" %tract_name)
	    #estimated_bundle_idx_ranked_med = estimated_bundle_idx_ranked[0:int(example_bundle_len_med)]
	    out_filename = '%s/%s/%s_%s_tract_afq5_0625_orig.tck' %(results_dir, sub, sub, tract_name)
	    #out_filename = '%s/%s/%s_%s_tract_wma5_0625_orig.tck' %(results_dir, sub, sub, tract_name)
	    #save_tract(estimated_bundle_idx_ranked_med, static_tractogram_Tractogram, static_tractogram, out_filename)

	    print("Estimating %s with fake LAP" %tract_name)
	    #estimated_bundle_idx_ranked_fake = estimated_bundle_idx_ranked[0:len_target]
	    out_filename = '%s/%s/%s_%s_tract_afq5_0625_fake.tck' %(results_dir, sub, sub, tract_name)
	    #out_filename = '%s/%s/%s_%s_tract_wma5_0625_fake.tck' %(results_dir, sub, sub, tract_name)
	    #save_tract(estimated_bundle_idx_ranked_fake, static_tractogram_Tractogram, static_tractogram, out_filename)

	    #plot ROC curve
	    print("Computing the ROC curve")
	    #true_tract_res = resample_tractogram(true_tract, step_size=0.625)
	    #true_tract = np.array(true_tract_res, dtype=np.object)
	    fpr, tpr, AUC = compute_roc_curve_lap(estimated_bundle_idx_ranked, true_tract, static_tractogram)
	    # Plot of the ROC curve
	    plt.interactive(True)
    	    plt.figure()
    	    lw = 1
    	    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' %AUC)
   	    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
   	    plt.xlim([0.0, 1.0])
   	    plt.ylim([0.0, 1.05])
  	    plt.xlabel('False Positive Rate')
  	    plt.ylabel('True Positive Rate')
  	    plt.title('ROC curve %s tract sub %s from 5 examples' %(tract_name, sub))
   	    plt.legend(loc="lower right")
   	    plt.show()

	

