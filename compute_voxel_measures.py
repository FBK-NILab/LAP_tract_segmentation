"""Compute DSC"""

from __future__ import print_function
import nibabel as nib
import numpy as np
import dipy
import scipy
from nibabel.streamlines import load, save 
from dipy.tracking.utils import affine_for_trackvis
from dipy.tracking.vox2track import streamline_mapping
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.life import voxel2streamline
import matplotlib.pyplot as plt


def compute_voxel_measures(estimated_tract, true_tract):

    #affine_true=affine_for_trackvis([1.25, 1.25, 1.25])
    aff=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])

    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()

    n_ET = len(estimated_tract)
    n_TT = len(true_tract)	
    dictionary_ET = streamline_mapping(estimated_tract, affine=aff)
    dictionary_TT = streamline_mapping(true_tract, affine=aff)
    voxel_list_intersection = set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract))

    sum_int_ET = 0
    sum_int_TT = 0
    for k in voxel_list_intersection:
	sum_int_ET = sum_int_ET + len(dictionary_ET[k])
	sum_int_TT = sum_int_TT + len(dictionary_TT[k])
    sum_int_ET = sum_int_ET/n_ET
    sum_int_TT = sum_int_TT/n_TT

    sum_ET = 0
    for k in voxel_list_estimated_tract:
	sum_ET = sum_ET + len(dictionary_ET[k])
    sum_ET = sum_ET/n_ET

    sum_TT = 0
    for k in voxel_list_true_tract:
	sum_TT = sum_TT + len(dictionary_TT[k])
    sum_TT = sum_TT/n_TT
    
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    FP = vol_B-TP
    FN = vol_A-TP
    sensitivity = float(TP) / float(TP + FN) 
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)
    wDSC = float(sum_int_ET + sum_int_TT) / float(sum_ET + sum_TT)
    J = float(TP) / float(TP + FN + FP)

    return DSC, wDSC, J, sensitivity


if __name__ == '__main__':

    sub = '992673'
    tract_name_list = ['Left_Arcuate']#, 'Callosum_Forceps_Minor', 'Right_Cingulum_Cingulate', 'Callosum_Forceps_Major']
    experiment =  'exp2' #'test'
    example_list = ['615744']#, '0006']#, '0007', '0008']
    true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment

    DSC_values = np.zeros((len(tract_name_list), len(example_list)))
    cost_values = np.zeros((len(tract_name_list), len(example_list)))

    for t, tract_name in enumerate(tract_name_list):
	
  	true_tract_filename = '%s/%s/%s_%s_tract.trk' %(true_tracts_dir, sub, sub, tract_name)
	true_tract = nib.streamlines.load(true_tract_filename)
	true_tract = true_tract.streamlines

    	for e, example in enumerate(example_list):
	
	    estimated_tract_filename = '%s/%s/%s_%s_tract_E%s.tck' %(results_dir, sub, sub, tract_name, example)
            estimated_tract = nib.streamlines.load(estimated_tract_filename)
	    estimated_tract = estimated_tract.streamlines	

	    DSC, wDSC, J, sensitivity = compute_voxel_measures(estimated_tract, true_tract)	
	    print("The DSC value is %s" %DSC)
	    print("The weighted DSC value is %s" %wDSC)
	    print("The Jaccard index is %s" %J)
	    print("The sensitivity is %s" %sensitivity)

	    result_lap = np.load('%s/%s/%s_%s_result_lap_E%s.npy' %(results_dir, sub, sub, tract_name, example))
	    min_cost_values = result_lap[1]
	    cost = np.sum(min_cost_values)/len(min_cost_values)

	    DSC_values[t,e] = DSC
	    cost_values[t,e] = cost

        #debugging
        DSC, wDSC, J, sensitivity = compute_voxel_measures(estimated_tract, estimated_tract)
        print("The DSC value is %s (must be 1)" %DSC)
	print("The weighted DSC value is %s (must be 1)" %wDSC)
	print("The Jaccard index is %s (must be 1)" %J)
	print("The sensitivity is %s (must be 1)" %sensitivity)
        DSC, wDSC, J, sensitivity = compute_voxel_measures(true_tract, true_tract)
        print("The DSC value is %s (must be 1)" %DSC)
	print("The weighted DSC value is %s (must be 1)" %wDSC)
	print("The Jaccard index is %s (must be 1)" %J)
	print("The sensitivity is %s (must be 1)" %sensitivity)


    #compute statistisc
    #DSC_vect = DSC_values.reshape((-1,))
    #cost_vect = cost_values.reshape((-1,))
    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(DSC_vect, cost_vect)
    #print("The R value is %s" %r_value)
    
    # plot
    #plt.interactive(True)
    #markers_list = ['o', '^', '*', 'd']
    #plt.figure()
    #for i in range(len(tract_name_list)):
    #    plt.scatter(DSC_values[i], cost_values[i], c='g',  marker=markers_list[i], s=70, label=tract_name_list[i])
    #plt.plot(DSC_vect, intercept + slope*DSC_vect, c='r', linestyle=':')
    #plt.xlabel("DSC")
    #plt.ylabel("cost")
    #plt.title('R = %s' %r_value)
    #plt.legend(loc=3)
    #plt.show()









