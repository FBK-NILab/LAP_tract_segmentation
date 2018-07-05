""" Combine different .tck files in a single .tck file using nibabel.
"""

import os
import sys
import argparse
import os.path
import nibabel as nib
import numpy as np
from nibabel.streamlines import load, save


def ensemble_tractograms(out_filename):
	
	cwd = os.getcwd()
	some_tractograms = []
	nb_streamlines = 0
	
	for filename in os.listdir(cwd):
		if filename.endswith(".tck"):
			tractogram = nib.streamlines.load(filename)
			tractogram = tractogram.streamlines
			nb_streamlines += len(tractogram)
			some_tractograms.append(tractogram)

	# Concatenate streamlines
	st=nib.streamlines.array_sequence.concatenate(some_tractograms[:], axis=0)

	# Retreiving header
	tractogram = nib.streamlines.load(filename)
	aff_vox_to_ras = tractogram.affine

	# Creating new header
	hdr = nib.streamlines.tck.TckFile.create_empty_header()
	hdr['voxel_to_rasmm'] = aff_vox_to_ras
	hdr['nb_streamlines'] = nb_streamlines

	# Saving ensemble tractogram
	et = nib.streamlines.tractogram.Tractogram(st, affine_to_rasmm=np.eye(4))
	nib.streamlines.save(et, out_filename, header=hdr)
	print("Ensemble tractogram saved in %s" % out_filename)
	

if __name__ == '__main__':

	np.random.seed(0) 

	parser = argparse.ArgumentParser()
	parser.add_argument('-out', nargs='?', const=1, default='',
	                    help='The output tractogram filename')                   
	args = parser.parse_args()

	ensemble_tractograms(args.out)

	sys.exit()
