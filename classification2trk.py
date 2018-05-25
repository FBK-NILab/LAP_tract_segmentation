"""Convert a classification.mat structure into a .tck tractogram."""

import scipy.io as sio
import numpy as np
import nibabel as nib
from lap_single_example import save_bundle


if __name__ == '__main__':

	# Load the classification structure
	matlabfile = sio.loadmat('index.mat')
	indeces = np.array(matlabfile['index'])

	out_filename = '720377_lobes_track_test.trk'
	
	t1 = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data/720337/anat/720337_t1.nii.gz'
	nii = nib.load(t1)
	tractogram_filename = '/N/u/gberto/Karst/Downloads/720377_track.tck'
	tractogram = nib.streamlines.load('/N/u/gberto/Karst/Downloads/720377_track.tck')
	tractogram = tractogram.streamlines

	print("Filtering tractogram...")
	idx_filtered_tractogram = []
	for i in range(len(tractogram)):
 		if indeces[i] > 2:
			idx_filtered_tractogram.append(i)

	#save_bundle(idx_filtered_tractogram, tractogram_filename, out_filename)
	
	filtered_tractogram = tractogram[idx_filtered_tractogram]
	
	# Creating header
	hdr = nib.streamlines.trk.TrkFile.create_empty_header()
	hdr['voxel_sizes'] = nii.header.get_zooms()[:3]
	hdr['voxel_order'] = 'LAS'
	hdr['dimensions'] = nii.shape[:3]
	hdr['voxel_to_rasmm'] = nii.affine

	# Saving tractogram
	print("Saving tractogram...")
	t = nib.streamlines.tractogram.Tractogram(filtered_tractogram, affine_to_rasmm=np.eye(4))
	nib.streamlines.save(t, out_filename, header=hdr)
	print("Tractogram saved in %s" % out_filename)


