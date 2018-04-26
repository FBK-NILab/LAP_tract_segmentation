from __future__ import print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt


if __name__ == '__main__':

	DSC_nn = np.load('DSC_nn_90.npy')
	DSC_rlap = np.load('DSC_rlap_90.npy')
	#DSC_nn = np.load('DSC_nn_R.npy')
	#DSC_rlap = np.load('DSC_rlap_R.npy')
	
	#delete the zero-values
	DSC_nn_masked = np.ma.masked_equal(DSC_nn, 0) 
	DSC_rlap_masked = np.ma.masked_equal(DSC_rlap, 0)  

	#compute the mean across the subjects: returns a 3D matrix
	DSC_mean_nn = np.mean(DSC_nn_masked, axis=(0,2))
	DSC_mean_rlap = np.mean(DSC_rlap_masked, axis=(0,2))
	DSC_std_nn = np.std(DSC_nn_masked, axis=(0,2))
	DSC_std_rlap = np.std(DSC_rlap_masked, axis=(0,2))
	
	#mean standard deviation
	ms=(np.mean(DSC_std_nn[:,:,:])+np.mean(DSC_std_rlap[:,:,:]))/2

	#standard deviation of the mean
	pairs=DSC_rlap.shape[0]**2-DSC_rlap.shape[0]
	sm=ms/np.sqrt(pairs)

	# PLOT

	from basic_units import cm, inch

	fig, ax = plt.subplots()
	
	#bundle_list = ['cstL', 'cstR', 'ifofL']
	bundle_list = ['cstL', 'cstR', 'ifofL', 'ifofR', 'thprefL', 'thprefR', 'ufL', 'ufR']
	N = DSC_rlap.shape[1]
	L = DSC_rlap.shape[3]
	ind = np.arange(N)    # the x locations for the groups
	width = 0.8/(2*L)     # the width of the bars 
	p1 = ax.bar(ind, DSC_mean_nn[:,0,0], width, color='r', bottom=0*cm, yerr=DSC_std_nn[:,0,0])
	p2 = ax.bar(ind + width, DSC_mean_rlap[:,0,0], width, color='c', bottom=0*cm, yerr=DSC_std_rlap[:,0,0])
	p3 = ax.bar(ind + 2*width, DSC_mean_nn[:,1,0], width, color='m', bottom=0*cm, yerr=DSC_std_nn[:,1,0])
	p4 = ax.bar(ind + 3*width, DSC_mean_rlap[:,1,0], width, color='g', bottom=0*cm, yerr=DSC_std_rlap[:,1,0])
	p5 = ax.bar(ind + 4*width, DSC_mean_nn[:,2,0], width, color='y', bottom=0*cm, yerr=DSC_std_nn[:,2,0])
	p6 = ax.bar(ind + 5*width, DSC_mean_rlap[:,2,0], width, color='b', bottom=0*cm, yerr=DSC_std_rlap[:,2,0])

	ax.set_title('Registration + segmentation comparison across %s pairs (std of the mean = %f)' %(pairs, sm))
	ax.set_xticks(ind + L*width)
	ax.set_xticklabels(bundle_list)
	ax.set_ylim(0,1)
	ax.legend((p1[0], p2[0], p3[0],  p4[0],  p5[0],  p6[0]), ('slr+NN', 'slr+RLAP', 'ant4t1w+NN', 'ant4t1w+RLAP', 'ant4fa+NN', 'ant4fa+RLAP'), loc=0)
	ax.set_ylabel('DSC')
	ax.autoscale_view()


	

	plt.show()	
