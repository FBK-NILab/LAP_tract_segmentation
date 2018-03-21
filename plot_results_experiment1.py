"""Plot the results of experiment 1."""

from __future__ import print_function
import nibabel as nib
import numpy as np
import dipy
import scipy
from nibabel.streamlines import load, save 
from compute_voxel_measures import compute_voxel_measures
import matplotlib.pyplot as plt


if __name__ == '__main__':

    experiment = 'exp1' 
    sub_list = ['983773', '990366', '991267', '993675', '996782']
    tract_name_list = ['Left_Arcuate', 'Callosum_Forceps_Minor']
    partition_list = ['A1', 'A4', 'A8', 'A12', 'A16']
    true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment

    DSC_values = np.zeros((len(sub_list), len(tract_name_list), len(partition_list)))
    
    for s, sub in enumerate(sub_list):

    	for t, tract_name in enumerate(tract_name_list):
	
  	    true_tract_filename = '%s/%s/%s_%s_tract.trk' %(true_tracts_dir, sub, sub, tract_name)
	    true_tract = nib.streamlines.load(true_tract_filename)
	    true_tract = true_tract.streamlines

    	    for p, partition in enumerate(partition_list):
	
	    	estimated_tract_filename = '%s/%s/%s_%s_tract_%s.tck' %(results_dir, sub, sub, tract_name, partition)
		estimated_tract = nib.streamlines.load(estimated_tract_filename)
   		estimated_tract = estimated_tract.streamlines	

	    	DSC, TP, vol_A, vol_B = compute_voxel_measures(estimated_tract, true_tract)	
	    	print("The DSC value is %s" %DSC)

	    	result_lap = np.load('%s/%s/%s_%s_result_lap_%s.npy' %(results_dir, sub, sub, tract_name, partition))
	    	DSC_values[s,t,p] = DSC

    #computing the mean across the sub
    DSC_values_mean = np.mean(DSC_values, axis=0)
    DSC_values_std = np.std(DSC_values, axis=0)


    # plot
    plt.interactive(True)
    plt.figure()
    color_list = ['g', 'r', 'y', 'b']
    markers_list = ['o', '^', '*', 'd']
    x = [1, 4, 8, 12, 16]
    for j in range(len(tract_name_list)):
	plt.errorbar(x, DSC_values_mean[j,:], yerr=DSC_values_std[j,:], c=color_list[j],  marker=markers_list[j],  label=tract_name_list[j])
    plt.xlabel("Number of examples")
    plt.ylabel("DSC")
    plt.title('Mean DSC across %s subjects' %len(sub_list))
    plt.legend(loc=4)
    plt.xlim(0, x[-1]+1)
    plt.show()


