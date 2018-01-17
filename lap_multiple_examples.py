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


def RLAP(kdt, k, dm_source_tract, source_tract, tractogram, distance):
    """Code for Rectangular Linear Assignment Problem.
    """
    tractogram = np.array(tractogram, dtype=np.object)
    D, I = kdt.query(dm_source_tract, k=k)
    superset = np.unique(I.flat)
    print("Computing the cost matrix (%s x %s) for RLAP... " % (len(source_tract),
                                                             len(superset)))
    cost_matrix = dissimilarity(source_tract, tractogram[superset], distance)
    print("Computing RLAP with LAPJV...")
    assignment = LinearAssignment(cost_matrix).solution
    estimated_bundle_idx = superset[assignment]
    min_cost_values = cost_matrix[np.arange(len(cost_matrix)), assignment]

    return estimated_bundle_idx, min_cost_values


def lap_multiple_examples(moving_tractograms_dir, static_tractogram, ex_dir, aff_dict, out_trk):
	"""Code for LAP from multiple examples.
	"""
	moving_tractograms_directory = os.listdir(moving_tractograms_dir)
	moving_tractograms_directory.sort()
	examples_directory = os.listdir(ex_dir)
	examples_directory.sort()

	nt = len(moving_tractograms_directory)
	ne = len(examples_directory)

	if nt != ne:
		print("Error: number of moving tractograms differs from number of example bundles.")
		sys.exit()
	else:	
		for i in range(nt):
			moving_tractogram = '%s/%s' %(moving_tractograms_dir, moving_tractograms_directory[i])
			example = '%s/%s' %(ex_dir, examples_directory[i])

			result_lap = lap_single_example(moving_tractogram, static_tractogram, example, aff_dict)

		return result_lap
		#return estimated_bundle


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