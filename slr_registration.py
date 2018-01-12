""" SLR (Streamline Linear Registration) of two tractograms.
    
    See Garyfallidis et. al, “Robust and efficient linear registration 
    of white-matter fascicles in the space of streamlines”, 
    Neuroimage, 117:124-140, 2015.
"""

import os
import sys
import argparse
import os.path
import nibabel as nib
import numpy as np
from nibabel.streamlines import load
from dipy.segment.clustering import QuickBundles
from dipy.align.streamlinear import StreamlineLinearRegistration
from dipy.tracking.streamline import set_number_of_points


def tractograms_slr(target_subject_id, source_subject_id, src_dir):
  
	target_tractogram_filename = '%s/sub-%s/sub-%s_var-FNAL_tract.trk' %(src_dir, target_subject_id, target_subject_id)  
	source_tractogram_filename = '%s/sub-%s/sub-%s_var-FNAL_tract.trk' %(src_dir, source_subject_id, source_subject_id)
	print("Target tractogram filename: %s" %target_tractogram_filename)
	print("Source tractogram filename: %s" %source_tractogram_filename)

	print("Loading tractograms...")
	target_tractogram = nib.streamlines.load(target_tractogram_filename)
	target_tractogram = target_tractogram.streamlines
	source_tractogram = nib.streamlines.load(source_tractogram_filename)
	source_tractogram = source_tractogram.streamlines                  

	print("Set parameters as in [Garyfallidis et al. 2015].") 
	threshold_length = 40.0 # 50mm / 1.25
	qb_threshold = 16.0  # 20mm / 1.25 
	nb_res_points = 20

	print("Performing QuickBundles of target tractogram and resampling...")
	tt = np.array([s for s in target_tractogram if len(s) > threshold_length], dtype=np.object)
	qb = QuickBundles(threshold=qb_threshold)
	tt_clusters = [cluster.centroid for cluster in qb.cluster(tt)]
	tt_clusters = set_number_of_points(tt_clusters, nb_res_points)

	print("Performing QuickBundles of source tractogram and resampling...")
	st = np.array([s for s in source_tractogram if len(s) > threshold_length], dtype=np.object)
	qb = QuickBundles(threshold=qb_threshold)
	st_clusters = [cluster.centroid for cluster in qb.cluster(st)]
	st_clusters = set_number_of_points(st_clusters, nb_res_points)

	print("Performing Linear Registration...")
	srr = StreamlineLinearRegistration()
	srm = srr.optimize(static=tt_clusters, moving=st_clusters)

	print("Affine transformation matrix with Streamline Linear Registration:")
	affine = srm.matrix
	print('%s' %affine)

	print("Applying the transformation...")
	source_tractogram_aligned = srm.transform(source_tractogram)

	return affine, source_tractogram_aligned


if __name__ == '__main__':

	# Setting the location of the repository
	script_src = os.path.basename(sys.argv[0]).strip('.py')
	script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
	src_dir = os.path.abspath(os.path.join(script_dir, '../data/derivatives/deterministic_tracking_dipy_FNAL'))
	#out_dir = os.path.join(src_dir, 'source_tractogram_aligned')
	#if not os.path.exists(out_dir):
	#    os.mkdir(out_dir)   

	parser = argparse.ArgumentParser()
	parser.add_argument('-target', nargs='?', const=1, default='',
	                    help='The target subject id')
	parser.add_argument('-source', nargs='?',  const=1, default='',
	                    help='The source subject id')
	                    
	args = parser.parse_args()

	affine, source_tractogram_aligned = tractograms_slr(args.target, args.source, src_dir)

	sys.exit()

	