"""Code for filter a tractogram given a classification matlab file.
Specifically here we are keeping all the streamlines that start and end in 
the two temporal, two occipital and two parietal lobes, which have indeces>2.
"""

import scipy.io as sio
import numpy as np
import nibabel as nib


def filter_tractogram_with_classification(tractogram_filename, classification, t1_filename):

	# Load the inputs
	tractogram = nib.streamlines.load(tractogram_filename)
	tractogram = tractogram.streamlines
	matlabfile = sio.loadmat(classification)
	indeces = np.array(matlabfile['index'])
	nii = nib.load(t1_filename)

	output_filename = tractogram[:-4] + '_filtered.trk'
	
	print("Filtering tractogram...")
	idx_filtered_tractogram = []
	for i in range(len(tractogram)):
 		if indeces[i] > 2:
			idx_filtered_tractogram.append(i)
	
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


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-tractogram', nargs='?', const=1, default='',
	                    help='The tractogram filename')
	parser.add_argument('-classification', nargs='?',  const=1, default='',
	                    help='The classification matlab structure filename')  
	parser.add_argument('-t1', nargs='?',  const=1, default='',
	                    help='The T1 filename')                  
	args = parser.parse_args()

	filter_tractogram_with_classification(args.tractogram, args.classification, args.t1)	
	                            
	sys.exit()    
