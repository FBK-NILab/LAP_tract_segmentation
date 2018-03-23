"""Plot the results of experiment 3 in comparison with results of experiment 1."""

from __future__ import print_function
import nibabel as nib
import numpy as np
import dipy
import scipy
from scipy.stats import linregress
from nibabel.streamlines import load, save 
from compute_voxel_measures import compute_voxel_measures
import matplotlib.pyplot as plt


if __name__ == '__main__':

    experiment_A = 'exp1' 
    experiment_B = 'exp3'
    sub_list = ['992673', '992774', '995174', '993675', '996782', '983773', '990366', '991267']
    tract_name_list = ['Left_Arcuate', 'Callosum_Forceps_Minor']
    partition_list_A = ['A8'] #['A1', 'A4', 'A8', 'A12', 'A16']
    partition_list_B = ['B8'] #['B1', 'B4', 'B8', 'B12', 'B16']
    true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir_A = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment_A
    results_dir_B = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment_B

    DSC_values_A = np.zeros((len(sub_list), len(tract_name_list), len(partition_list_A)))
    DSC_values_B = np.zeros((len(sub_list), len(tract_name_list), len(partition_list_B)))
    
    for s, sub in enumerate(sub_list):

    	for t, tract_name in enumerate(tract_name_list):
	
  	    true_tract_filename = '%s/%s/%s_%s_tract.trk' %(true_tracts_dir, sub, sub, tract_name)
	    true_tract = nib.streamlines.load(true_tract_filename)
	    true_tract = true_tract.streamlines

    	    for p, partition_A in enumerate(partition_list_A):
	   	
		#partition A
	    	estimated_tract_filename = '%s/%s/%s_%s_tract_%s.tck' %(results_dir_A, sub, sub, tract_name, partition_A)
		estimated_tract = nib.streamlines.load(estimated_tract_filename)
   		estimated_tract = estimated_tract.streamlines	

	    	DSC, TP, vol_A, vol_B = compute_voxel_measures(estimated_tract, true_tract)	
	    	print("The DSC value is %s" %DSC)
	    	DSC_values_A[s,t,p] = DSC

		#partition B
		partition_B = partition_list_B[p]
	    	estimated_tract_filename = '%s/%s/%s_%s_tract_%s.tck' %(results_dir_B, sub, sub, tract_name, partition_B)
		estimated_tract = nib.streamlines.load(estimated_tract_filename)
   		estimated_tract = estimated_tract.streamlines	

	    	DSC, TP, vol_A, vol_B = compute_voxel_measures(estimated_tract, true_tract)	
	    	print("The DSC value is %s" %DSC)
	    	DSC_values_B[s,t,p] = DSC


    # plot
    plt.interactive(True)
    color_list = ['g', 'r', 'y', 'b', 'k', 'm', 'c', 'p']
    markers_list = ['o', '^', '*', 'd']

    for k in range(len(partition_list_A)):
	#reshape arrays
        DSC_vect_A = DSC_values_A[:,:,k].reshape((-1,))
        DSC_vect_B = DSC_values_B[:,:,k].reshape((-1,))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(DSC_vect_A, DSC_vect_B)
        print("The R value is %s" %r_value)

    	plt.figure(k)
    	for i in range(len(sub_list)):
            for j in range(len(tract_name_list)):
	        plt.scatter(DSC_values_A[i,j,k], DSC_values_B[i,j,k], c=color_list[j],  marker=markers_list[j], s=70)
		if (i==0):
		    plt.scatter(DSC_values_A[i,j,k], DSC_values_B[i,j,k], c=color_list[j],  marker=markers_list[j], s=70, label=tract_name_list[j])
        plt.plot(DSC_vect_A, intercept + slope*DSC_vect_A, c='r', linestyle=':')
    	plt.xlabel("DSC partition %s" %partition_list_A[k])
    	plt.ylabel("DSC partition %s" %partition_list_B[k])
    	plt.title('Does the choice of the examples matter? R = %0.3f' %r_value)
    	plt.legend(loc=4, fontsize='small')
    plt.show()




