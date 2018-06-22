from __future__ import division
import nibabel as nib
from nibabel.streamlines import load
import numpy as np
from dipy.tracking.metrics import endpoint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def compute_endpoints(bundle):
	endpoints = np.zeros((len(bundle),3)) 
	for i, st in enumerate(bundle):
		endpoints[i] = endpoint(st)
	return endpoints		


def compute_startpoints(bundle):
	bundle_flip = []
	for st in bundle:
		tmp = np.flip(st, axis=0)
		bundle_flip.append(tmp)
	startpoints = compute_endpoints(bundle_flip)
	return startpoints	


def orient_tract(bundle, y_th=None, z_th=None):
	oriented_bundle = []
	if y_th:
		for st in bundle:	
			if st[0][1] < y_th:
				tmp = np.flip(st, axis=0)
				oriented_bundle.append(tmp)
	elif z_th:
		for st in bundle:	
			if st[0][2] < z_th:
				tmp = np.flip(st, axis=0)
				oriented_bundle.append(tmp)
	else:
		oriented_bundle = bundle
	return oriented_bundle	


def plot_terminal_points(bundle):
	endpoints = compute_endpoints(bundle)
	startpoints = compute_startpoints(bundle)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	xs = endpoints[:,0]
	ys = endpoints[:,1]
	zs = endpoints[:,2]
	xsf = startpoints[:,0]
	ysf = startpoints[:,1]
	zsf = startpoints[:,2]
	ax.scatter(xs, ys, zs, c='b', marker='o')
	ax.scatter(xsf, ysf, zsf, c='r', marker='o')
	ax.set_xlabel('x label')
	ax.set_ylabel('y label')
	ax.set_zlabel('z label')
	#plt.title("%s" %tract_filename)
	plt.show()


if __name__ == '__main__':

    # Load data
    #tract = nib.streamlines.load('/home/giulia/prni2017_streamline_distances/code/data/100307/wmql_tracts/100307_ifof.right.trk')
    
	tract_list = ['/home/giulia/Downloads/910241_Right_MdLF-Ang_tract.trk',
    			  '/home/giulia/Downloads/910241_Right_TPC_tract.trk',
    			  '/home/giulia/Downloads/910241_Left_MdLF-SPL_tract.trk',
    			  '/home/giulia/Downloads/910241_Right_pArc_tract.trk',
    			  '/home/giulia/Downloads/910241_Right_Thalamic_Radiation_tract.trk',
    			  '/home/giulia/Downloads/910241_Right_Arcuate_tract.trk',
   				  '/home/giulia/Downloads/910241_Right_Corticospinal_tract.trk',
   				  '/home/giulia/Downloads/910241_Left_IFOF_tract.trk']

	y_th_list = [-14.5, None, -25, None, None, None, None, None]
	z_th_list = [None, 12, None, 16, None, None, None, None]

	for t, tract_name in enumerate(tract_list):
 		tract = nib.streamlines.load(tract_name)
		tract = tract.streamlines
		oriented_tract = orient_tract(tract, y_th_list[t], z_th_list[t]) 
		plot_terminal_points(oriented_tract)

   