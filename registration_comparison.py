from __future__ import print_function, division
import numpy as np

from __future__ import print_function
import nibabel as nib
import numpy as np
from nibabel.streamlines import load, save
from lap_single_example import compute_kdtree_and_dr_tractogram, RLAP, save_bundle
from dipy.tracking.distances import bundles_distances_mam
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


	
	print("Compute the dissimilarity of the aligned example bundle with the prototypes of target tractogram.")
	moving_example = np.array(moving_example, dtype=np.object)
	dm_moving_example = distance_func(moving_example, prototypes)

	print("Segmentation as Rectangular linear Assignment Problem (RLAP).")
	estimated_bundle_idx, min_cost_values = RLAP(kdt, k, dm_moving_example, moving_example, static_tractogram, distance_func)
	#estimated_bundle = static_tractogram[estimated_bundle_idx]


def NN(kdt, dm_source_tract):
    """Code for efficient approximate nearest neighbors computation.
    """
    D, I = kdt.query(dm_source_tract, k=1)
    return I.squeeze()



if __name__ == '__main__':

	sub_list = [100307, 109123, 131217, 199655, 341834, 599671, 601127, 756055, 770352, 917255]
	distance_func_list = ['bundles_distances_mam']
	bundle_list = ['cbL', 'cbR', 'cstL', 'cstR', 'ifofL', 'ifofR', 'thprefL', 'thprefR', 'ufL', 'ufR']
	method = ['nn', 'rlap']
	registration = ['slr', 'ant4fa', 'ant4t1w']
	k = 500

	basedir = 'miccai2018_dataset'

	for ss, static_sub in enumerate(sub_list):
		tractogram_dir = 'deterministic_tracking_dipy_FNAL'
		static_tractogram_filename = '%s/%s/sub-%s/sub-%s_var-FNAL_tract.trk' %(basedir, tractogram_dir, static_sub, static_sub)
		static_tractogram = nib.streamlines.load(static_tractogram_filename)
		static_tractogram = static_tractogram.streamlines
		
		for d, dist in enumerate(distance_func_list):
			print("Compute the dissimilarity representation of the target tractogram and build the kd-tree.")
			kdt, prototypes = compute_kdtree_and_dr_tractogram(static_tractogram, distance=dist)

			for ms, moving_sub in enumerate(sub_list):
				if moving_sub != static_sub:
					for b, bundle in enumerate(bundle_list):
						for m, met in enumerate(method):
							for r, reg in enumerate(registration):
								if reg == 'slr':
									slr_dir = basedir + 'streamline_based_affine_registration'
									bundle_filename = '%s/sub-%s/sub-%s_var-slr_space-%s_set-%s_tract.trk' %(slr_dir, moving_sub, moving_sub, static_sub, bundle)
								else:
									ants_dir = basedir + 'voxel_based_registration'
									bundle_filename = '%s/sub-%s/sub-%s_space_%s_var-%s_set-%s_tract.trk' %(ants_dir, moving_sub, moving_sub, static_sub, reg, bundle)
	
