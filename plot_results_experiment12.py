"""Plot results exp12."""

from __future__ import print_function
import nibabel as nib
import numpy as np
import dipy
import os
from compute_voxel_measures import compute_voxel_measures
import matplotlib.pyplot as plt

def print_latex_table(matrix2D, column_labels, row_labels):
	print("\\begin{table} [h!]")
	print("\\centering")
	tmp = ' | '.join('c' * (len(column_labels) + 1))
	print("\\begin{tabular}{ %s }" % tmp)
	print(' & ' + ' & '.join([col for col in column_labels]) + ' \\\\')
	print('\\hline')
	print('\\hline')
	for i, row in enumerate(matrix2D):
		print('\t ' + row_labels[i] + ' & ' + ' & '.join(["%.2f" % v for v in matrix2D[i]]) + ' \\\\')
		print('\t \\hline')	
	print("\\end{tabular}")
	print("\\caption{\small{Add caption here.}}")
	print("\\end{table}\n")


if __name__ == '__main__':

    experiment = 'exp12'
    sub_list = ['910241', '910443', '911849', '983773', '991267']#, '912447', '917255', '989987']
    #tract_name_list = ['Left_Thalamic_Radiation', 'Right_Thalamic_Radiation', 'Left_Corticospinal', 'Right_Corticospinal', 'Left_IFOF', 'Right_IFOF', 'Left_Arcuate', 'Right_Arcuate']
    tract_name_list = ['Left_pArc', 'Right_pArc', 'Left_TPC', 'Right_TPC', 'Left_MdLF-SPL', 'Right_MdLF-SPL', 'Left_MdLF-Ang', 'Right_MdLF-Ang']
    #true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk_ens_prob_afq'
    true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk_ens_prob_wma'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment

    matrix_original = np.zeros((len(sub_list), len(tract_name_list)))
    matrix_fake = np.zeros((len(sub_list), len(tract_name_list)))

    matrix_len_original = np.zeros((len(sub_list), len(tract_name_list)))
    matrix_len_true = np.zeros((len(sub_list), len(tract_name_list)))

    for s, sub in enumerate(sub_list):

    	for t, tract_name in enumerate(tract_name_list):
	
  	    true_tract_filename = '%s/%s/%s_%s_tract.trk' %(true_tracts_dir, sub, sub, tract_name)
	    true_tract = nib.streamlines.load(true_tract_filename)
	    true_tract = true_tract.streamlines

	    #DSC original
	    #original_tract_filename = '%s/%s/%s_%s_tract_afq5_0625_orig.tck' %(results_dir, sub, sub, tract_name)
	    original_tract_filename = '%s/%s/%s_%s_tract_wma5_0625_orig.tck' %(results_dir, sub, sub, tract_name)
	    original_tract = nib.streamlines.load(original_tract_filename)
	    original_tract = original_tract.streamlines
	    
	    DSC, wDSC, J, sensitivity, vol_A, vol_B = compute_voxel_measures(original_tract, true_tract)
	    print("For %s the DSC with original LAP is %0.3f (%s streamlines)" %(tract_name, DSC, len(original_tract)))
	    matrix_original[s,t] = DSC
	    matrix_len_original[s,t] = len(original_tract)

	    #DSC fake
	    #fake_tract_filename = '%s/%s/%s_%s_tract_afq5_0625_fake.tck' %(results_dir, sub, sub, tract_name)
	    fake_tract_filename = '%s/%s/%s_%s_tract_wma5_0625_fake.tck' %(results_dir, sub, sub, tract_name)
	    fake_tract = nib.streamlines.load(fake_tract_filename)
	    fake_tract = fake_tract.streamlines

	    DSC, wDSC, J, sensitivity, vol_A, vol_B = compute_voxel_measures(fake_tract, true_tract)
	    print("For %s the DSC with fake LAP is %0.3f (%s streamlines)" %(tract_name, DSC, len(fake_tract)))
	    matrix_fake[s,t] = DSC
	    matrix_len_true[s,t] = len(fake_tract)

    #Combine the two matrices for visualization
    summary_matrix = np.zeros((2*len(tract_name_list), len(sub_list)))
    for row in range(2*len(tract_name_list)):
	if np.mod(row,2) == 0:
	    summary_matrix[row] = matrix_original[:, row/2]
	else:
	    summary_matrix[row] = matrix_fake[:, (row-1)/2]

    #row_labels = ['TRl', 'fake', 'TRr', 'fake', 'CSTl', 'fake', 'CSTr', 'fake', 'IFOFl', 'fake', 'IFOFr', 'fake', 'AFl', 'fake', 'AFr', 'fake']
    row_labels = ['pArcl', 'fake', 'pArcr', 'fake', 'TPCl', 'fake', 'TPCr', 'fake', 'MdLF-SPLl', 'fake', 'MdLF-SPLr', 'fake', 'MdLF-Angl', 'fake', 'MdLF-Angr', 'fake']
    print_latex_table(summary_matrix, sub_list, row_labels)


    # PLOT
    plt.interactive(True)
    color_list = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y']
    markers_list = ['o','o', '^','^', '*', '*', 'd','d']


    #PLOT delta DSC vs delta length
    #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_length, DSC_vect)
    #print("The R value is %s" %r_value)
    
    delta_dsc = matrix_fake - matrix_original
    delta_length = matrix_len_true - matrix_len_original

    #plt.figure()
    for j in range(len(tract_name_list)):
        plt.scatter(delta_length[:,j], delta_dsc[:,j], c='y', marker=markers_list[j], s=70, label=tract_name_list[j])
    #plt.plot(target_length, intercept + slope*target_length, c='r', linestyle=':')
    plt.xlabel("number of streamlines true tract - number of streamlines original lap (median) ", fontsize=16)
    plt.ylabel("DSC fake lap - DSC original lap", fontsize=16)
    #plt.xlim([0, np.max(delta_length)+100])
    #plt.ylim([0.4,0.9])
    plt.axhline(y=0, color='r', linestyle='-')
    plt.axvline(x=0, color='r', linestyle='-')
    plt.title('Left side: overestimation. Right side: underestimation.', fontsize=16)
    plt.legend(loc=0, fontsize='small')

    plt.show()
  
