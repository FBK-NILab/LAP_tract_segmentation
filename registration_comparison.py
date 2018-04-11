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

	sub_list = ['100307', '109123']#[100307, 109123, 131217, 199655, 341834, 599671, 601127, 756055, 770352, 917255]
	distance_list = ['mam']#['mam', 'varifolds']
	bundle_list = ['ifofL']#['cbL', 'cbR', 'cstL', 'cstR', 'ifofL', 'ifofR', 'thprefL', 'thprefR', 'ufL', 'ufR']
	registration = ['slr']#['slr', 'ant4fa', 'ant4t1w']
	method = ['rlap']#['nn', 'rlap']
	k = 500

	basedir = 'miccai2018_dataset/'
	DSC_mam_nn = np.zeros(len(sub_list), len(sub_list), len(bundle_list), len(registration))
	DSC_mam_nn = np.zeros(len(sub_list), len(sub_list), len(bundle_list), len(registration))

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

			for ms, moving_sub in enumerate(sub_list):
				if moving_sub != static_sub:
					for b, bundle in enumerate(bundle_list):
						for r, reg in enumerate(registration):
							if reg == 'slr':
								slr_dir = basedir + 'streamline_based_affine_registration'
								bundle_filename = '%s/sub-%s/sub-%s_var-slr_space-%s_set-%s_tract.trk' %(slr_dir, moving_sub, moving_sub, static_sub, bundle)
							else:
								ants_dir = basedir + 'voxel_based_registration'
								bundle_filename = '%s/sub-%s/sub-%s_space_%s_var-%s_set-%s_tract.trk' %(ants_dir, moving_sub, moving_sub, static_sub, reg, bundle)
							for m, met in enumerate(method):
								if met == 'rlap':
									example_bundle = nib.streamlines.load(bundle_filename)
									example_bundle = example_bundle.streamlines
									print("Compute the dissimilarity of the aligned example bundle with the prototypes of static tractogram.")
									moving_example = np.array(example_bundle, dtype=np.object)
									dm_moving_example = distance_func(moving_example, prototypes)
									print("Segmentation as Rectangular linear Assignment Problem (RLAP).")
									estimated_bundle_idx, min_cost_values = RLAP(kdt, k, dm_moving_example, moving_example, static_tractogram, distance_func)
									#estimated_bundle = static_tractogram[estimated_bundle_idx]
