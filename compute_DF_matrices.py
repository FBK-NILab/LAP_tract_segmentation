from __future__ import print_function
import nibabel as nib
import numpy as np
from nibabel.streamlines import load, save
from lap_single_example import compute_kdtree_and_dr_tractogram
from dipy.tracking.distances import bundles_distances_mam
from compute_voxel_measures import compute_voxel_measures
from sklearn.neighbors import KDTree
from dissimilarity import compute_dissimilarity, dissimilarity
from dipy.tracking.streamline import length, set_number_of_points
from test_shape import frenet_diff
import time


def compute_lap_inputs(kdt, k, dm_source_tract, source_tract, tractogram, distance, nbp=12):
    """Code for computing the inputs to the Rectangular Linear Assignment Problem.
    """
    tractogram = np.array(tractogram, dtype=np.object)
    D, I = kdt.query(dm_source_tract, k=k)
    superset = np.unique(I.flat)
    np.save('superset_idx', superset)
    print("Computing the distance matrix (%s x %s) for RLAP... " % (len(source_tract),
                                                             len(superset)))
    t0=time.time()
    distance_matrix = dissimilarity(source_tract, tractogram[superset], distance)
    print("Time for computing the distance matrix = %s seconds" %(time.time()-t0))
    print("Computing the bundle similarity matrix (%s x %s) for RLAP... " % (len(source_tract),
                                                             len(superset)))
    t1=time.time()
    frenet_matrix = frenet_diff(source_tract, tractogram[superset], nbp)
    print("Time for computing the shape matrix = %s seconds" %(time.time()-t1))

    return distance_matrix, frenet_matrix, superset



if __name__ == '__main__':

	sub_list = ['100307', '109123', '131217']#, '199655', '341834']#, '599671', '601127', '756055', '770352', '917255']
	distance_list = ['mam']#['mam', 'varifolds']
	bundle_list = ['ufL', 'cstL']#, 'ufR', 'cstL', 'cstR', 'ifofL', 'ifofR', 'thprefL', 'thprefR']
	registration = ['slr']#'ant4t1w', 'ant4fa']
	method = ['rlap_frenet'] #rlap
	h_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
	k = 500

	np.random.seed(0)

	basedir = 'miccai2018_dataset'
	#DSC_nn = np.zeros((len(sub_list), len(bundle_list), len(sub_list), len(registration), len(h_list)))
	#DSC_rlap = np.zeros((len(sub_list), len(bundle_list), len(sub_list), len(registration), len(h_list)))
	DSC_rlap_frenet = np.zeros((len(sub_list), len(bundle_list), len(sub_list), len(registration), len(h_list)))

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
								if met == 'rlap_frenet':
									print("Segmentation as MODIFIED WITH FRENET Rectangular linear Assignment Problem (RLAP).")
									distance_matrix, frenet_matrix, superset = compute_lap_inputs(kdt, k, dm_moving_example, moving_example, static_tractogram, distance_func)
									d_matrix_name = 'D_m%s_s%s_set-%s.npy' %(moving_sub, static_sub, bundle)
									f_matrix_name = 'F_m%s_s%s_set-%s.npy' %(moving_sub, static_sub, bundle)
									superset_name = 'superset_m%s_s%s_set-%s.npy' %(moving_sub, static_sub, bundle)
									np.save(d_matrix_name, distance_matrix)
									np.save(f_matrix_name, frenet_matrix)
									np.save(superset_name, superset)
									
									
