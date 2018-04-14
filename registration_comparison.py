from __future__ import print_function
import nibabel as nib
import numpy as np
from nibabel.streamlines import load, save
from lap_single_example import compute_kdtree_and_dr_tractogram, RLAP, save_bundle
from dipy.tracking.distances import bundles_distances_mam
from compute_voxel_measures import compute_voxel_measures
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


def NN(kdt, dm_source_tract):
    """Code for efficient approximate nearest neighbors computation.
    """
    D, I = kdt.query(dm_source_tract, k=1)
    return I.squeeze()



if __name__ == '__main__':

	sub_list = ['100307', '109123', '131217', '199655']#[100307, 109123, 131217, 199655, 341834, 599671, 601127, 756055, 770352, 917255]
	distance_list = ['mam']#['mam', 'varifolds']
	bundle_list = ['ifofL', 'thprefL', 'cstL']#['cbL', 'cbR', 'cstL', 'cstR', 'ifofL', 'ifofR', 'thprefL', 'thprefR', 'ufL', 'ufR']
	registration = ['slr', 'ant4t1w', 'ant4fa']
	method = ['nn', 'rlap']
	k = 500

	basedir = 'miccai2018_dataset'
	DSC_nn = np.zeros((len(sub_list), len(bundle_list), len(sub_list), len(registration), len(distance_list)))
	DSC_rlap = np.zeros((len(sub_list), len(bundle_list), len(sub_list), len(registration), len(distance_list)))

	for ss, static_sub in enumerate(sub_list):
		tractogram_dir = 'deterministic_tracking_dipy_FNAL'
		static_tractogram_filename = '%s/%s/sub-%s/sub-%s_var-FNAL_tract.trk' %(basedir, tractogram_dir, static_sub, static_sub)
		static_tractogram = nib.streamlines.load(static_tractogram_filename)
		static_tractogram = static_tractogram.streamlines
		
		for d, dist in enumerate(distance_list):
			if dist == 'mam':
				distance_func = bundles_distances_mam
			elif dist == 'varifolds':
				distance_func = varifolds
			else:
				print("Distance %s not supported yet." % dist)
        		Exception		 	
			print("Compute the dissimilarity representation of the static tractogram and build the kd-tree.")
			kdt, prototypes = compute_kdtree_and_dr_tractogram(static_tractogram, distance=distance_func)

			for b, bundle in enumerate(bundle_list):
				wmql_dir = 'wmql_FNALW' 
				true_bundle_filename = '%s/%s/sub-%s/sub-%s_var-FNALW_set-%s_tract.trk' %(basedir, wmql_dir, static_sub, static_sub, bundle)
				true_bundle = nib.streamlines.load(true_bundle_filename)
				true_bundle = true_bundle.streamlines

				for ms, moving_sub in enumerate(sub_list):
					if moving_sub != static_sub:
						for r, reg in enumerate(registration):
							if reg == 'slr':
								slr_dir = basedir + '/streamline_based_affine_registration'
								bundle_filename = '%s/sub-%s/sub-%s_var-slr_space-%s_set-%s_tract.trk' %(slr_dir, moving_sub, moving_sub, static_sub, bundle)
							else:
								ants_dir = basedir + '/voxel_based_registration'
								bundle_filename = '%s/sub-%s/sub-%s_space_%s_var-%s_set-%s_tract.trk' %(ants_dir, moving_sub, moving_sub, static_sub, reg, bundle)
							example_bundle = nib.streamlines.load(bundle_filename)
							example_bundle = example_bundle.streamlines
							print("Compute the dissimilarity of the aligned example bundle with the prototypes of static tractogram.")
							moving_example = np.array(example_bundle, dtype=np.object)
							dm_moving_example = distance_func(moving_example, prototypes)

							for m, met in enumerate(method):
								if met == 'rlap':
									print("Segmentation as Rectangular linear Assignment Problem (RLAP).")
									estimated_bundle_idx, min_cost_values = RLAP(kdt, k, dm_moving_example, moving_example, static_tractogram, distance_func)
									estimated_bundle = static_tractogram[estimated_bundle_idx]
									print("Computing the DSC value.")
									DSC, TP, vol_A, vol_B = compute_voxel_measures(estimated_bundle, true_bundle)	
									print("The DSC value is %s" %DSC)
									DSC_rlap[ss, b, ms, r, d] = DSC
								if met == 'nn':
									print("Segmentation as Nearest Neighbor.")	
									estimated_bundle_idx = NN(kdt, dm_moving_example)
									estimated_bundle = static_tractogram[estimated_bundle_idx]
									print("Computing the DSC value.")
									DSC, TP, vol_A, vol_B = compute_voxel_measures(estimated_bundle, true_bundle)
									print("The DSC value is %s" %DSC)																																																																																																																																																								
									DSC_nn[ss, b, ms, r, d] = DSC


	np.save('DSC_nn', DSC_nn)
	np.save('DSC_rlap', DSC_rlap)
	
	#delete the zero-values
	DSC_nn_masked = np.ma.masked_equal(DSC_nn, 0) 
	DSC_rlap_masked = np.ma.masked_equal(DSC_rlap, 0)  

	#compute the mean across the subjects: returns a 3D matrix
	DSC_mean_nn = np.mean(DSC_nn_masked, axis=(0,2))
	DSC_mean_rlap = np.mean(DSC_rlap_masked, axis=(0,2))
	DSC_std_nn = np.std(DSC_nn_masked, axis=(0,2))
	DSC_std_rlap = np.std(DSC_rlap_masked, axis=(0,2))


	# PLOT

	from basic_units import cm, inch

	fig, ax = plt.subplots()

	ind = np.arange(3)    # the x locations for the groups
	width = 0.3        # the width of the bars
	p1 = ax.bar(ind, DSC_mean_nn[:,0,0][0], width, color='r', bottom=0*cm, yerr=DSC_std_nn[:,0,0][0])
	p2 = ax.bar(ind + width, DSC_mean_nn[:,1,0][0], width, color='y', bottom=0*cm, yerr=DSC_std_nn[:,1,0][0])
	p3 = ax.bar(ind + 2*width, DSC_mean_nn[:,2,0][0], width, color='g', bottom=0*cm, yerr=DSC_std_nn[:,2,0][0])

	ax.set_title('Registration comparison with NN')
	ax.set_xticks(ind + 3*width / 2)
	ax.set_xticklabels(('ifofL', 'thprefL', 'cstL'))

	ax.legend((p1[0], p2[0], p3[0]), ('slr', 'ant4fa', 'ant4t1w'))
	ax.yaxis.set_units('DSC')
	ax.autoscale_view()

	plt.show()							


