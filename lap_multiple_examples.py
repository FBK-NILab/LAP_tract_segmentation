""" Bundle segmentation with Rectangular Linear Assignment Problem.

	See Sharmin et al., 'White Matter Tract Segmentation as Multiple 
	Linear Assignment Problems', Fronts. Neurosci., 2017.
"""

import os
import sys
import argparse
import os.path
import nibabel as nib
import numpy as np
from nibabel.streamlines import load, save
from tractograms_slr import tractograms_slr
from lap_single_example import lap_single_example

try:
    from joblib import Parallel, delayed
    joblib_available = True
except ImportError:
    joblib_available = False


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


def lap_multiple_examples(moving_tractograms_dir, static_tractogram, ex_dir, aff_dict, out_trk):
	"""Code for LAP from multiple examples.
	"""
	moving_tractograms = os.listdir(moving_tractograms_dir)
	moving_tractograms.sort()
	examples = os.listdir(ex_dir)
	examples.sort()

	nt = len(moving_tractograms)
	ne = len(examples)

	if nt != ne:
		print("Error: number of moving tractograms differs from number of example bundles.")
		sys.exit()
	else:	
		result_lap = []
		for i in range(nt):
			moving_tractogram = '%s/%s' %(moving_tractograms_dir, moving_tractograms[i])
			example = '%s/%s' %(ex_dir, examples[i])
			tmp = np.array([lap_single_example(moving_tractogram, static_tractogram, example, aff_dict)])
			result_lap.append(tmp)

		result_lap = np.array(result_lap)
		estimated_bundle_idx = np.hstack(result_lap[:,0,0])
		min_cost_values = np.hstack(result_lap[:,0,1])
		example_bundle_len_med = np.median(np.hstack(result_lap[:,0,2]))

		print("Ranking the estimated streamlines...")
		estimated_bundle_idx_ranked = ranking_schema(estimated_bundle_idx, min_cost_values)                                                       

		print("Extracting the estimated bundle...")
		estimated_bundle_idx_ranked_med = estimated_bundle_idx_ranked[0:int(example_bundle_len_med)]
		static_tractogram = nib.streamlines.load(static_tractogram)
		aff_vox_to_ras = static_tractogram.affine
		voxel_sizes = static_tractogram.header['voxel_sizes']
		dimensions = static_tractogram.header['dimensions']
		static_tractogram = static_tractogram.streamlines
		estimated_bundle = static_tractogram[estimated_bundle_idx_ranked_med]

		# Creating header
		hdr = nib.streamlines.trk.TrkFile.create_empty_header()
		hdr['voxel_sizes'] = voxel_sizes
		hdr['voxel_order'] = 'LAS'
		hdr['dimensions'] = dimensions
		hdr['voxel_to_rasmm'] = aff_vox_to_ras 

		# Saving tractogram
		t = nib.streamlines.tractogram.Tractogram(estimated_bundle, affine_to_rasmm=np.eye(4))
		print("Saving tractogram in %s" % out_trk)
		nib.streamlines.save(t, out_trk , header=hdr)
		print("Tractogram saved in %s" % out_trk)

		return estimated_bundle


if __name__ == '__main__':

	np.random.seed(0) 

	parser = argparse.ArgumentParser()
	parser.add_argument('-moving_dir', nargs='?', const=1, default='',
	                    help='The moving tractogram directory')
	parser.add_argument('-static', nargs='?',  const=1, default='',
	                    help='The static tractogram filename')
	parser.add_argument('-ex_dir', nargs='?',  const=1, default='',
	                    help='The examples bundle directory')
	parser.add_argument('-aff', nargs='?',  const=1, default='',
	                    help='The input affine table filename')
	parser.add_argument('-out', nargs='?',  const=1, default='default',
	                    help='The output estimated bundle filename')                   
	args = parser.parse_args()

	estimated_bundle = lap_multiple_examples(args.moving_dir, args.static, args.ex_dir, args.aff, args.out)

	sys.exit()    