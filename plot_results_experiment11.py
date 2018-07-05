"""Plot the results of experiment 11 that ran on BL."""

from __future__ import print_function
import nibabel as nib
import numpy as np
import dipy
import scipy
import scipy.stats
from scipy.io import loadmat
import matplotlib.pyplot as plt


if __name__ == '__main__':

    experiment = 'exp11'
    sub_list = ['910241', '910443', '911849']#['910241','910443']#
    example_list = ['500222', '506234', '510225']#['510225', '506234']#
    tract_name_list = ['Left_Thalamic_Radiation', 'Right_Thalamic_Radiation', 'Left_Corticospinal', 'Right_Corticospinal', 'Left_IFOF', 'Right_IFOF', 'Left_Arcuate', 'Right_Arcuate', 'Left_pArc', 'Right_pArc', 'Left_TPC', 'Right_TPC', 'Left_MdLF-SPL', 'Right_MdLF-SPL', 'Left_MdLF-Ang', 'Right_MdLF-Ang']
    #true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    #examples_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment

    #arrays for AFQ
    target_count_afq = np.zeros((len(sub_list),8,3))
    results_afq = np.zeros((len(sub_list), len(example_list), 8, 6))
    for s, sub in enumerate(sub_list):
	target_count_filename = '%s/%s_0.2/%s_afq_output_counts.mat' %(results_dir, sub, sub)
	data = scipy.io.loadmat(target_count_filename)
	target_count_afq[s,:,:] = np.array(data['output_counts'])

    	for e, example in enumerate(example_list):
	    results_afq_filename = '%s/%s_0.2/%s_results_afq_E%s.npy' %(results_dir, sub, sub, example)
	    results_afq[s,e,:,:] = np.load(results_afq_filename)

    #arrays for WMA
    target_count_wma = np.zeros((len(sub_list),8,3))
    results_wma = np.zeros((len(sub_list), len(example_list), 8, 6))
    for s, sub in enumerate(sub_list):
	target_count_filename = '%s/%s_0.2/%s_wma_output_counts.mat' %(results_dir, sub, sub)
	data = scipy.io.loadmat(target_count_filename)
	target_count_wma[s,:,:] = np.array(data['output_counts'])

    	for e, example in enumerate(example_list):
	    results_wma_filename = '%s/%s_0.2/%s_results_wma_E%s.npy' %(results_dir, sub, sub, example)
	    results_wma[s,e,:,:] = np.load(results_wma_filename)

    #mean on example subjects
    results_afq_m = np.mean(results_afq, axis=1)
    DSC_afq_vect = results_afq_m[:,:,0].reshape((-1,))
    wDSC_afq_vect = results_afq_m[:,:,1].reshape((-1,))

    results_wma_m = np.mean(results_wma, axis=1)
    DSC_wma_vect = results_wma_m[:,:,0].reshape((-1,))
    wDSC_wma_vect = results_wma_m[:,:,1].reshape((-1,))

    #concatenate vectors AFQ + WMA
    DSC_vect = np.concatenate((DSC_afq_vect, DSC_wma_vect))
    wDSC_vect = np.concatenate((wDSC_afq_vect, wDSC_wma_vect))
    target_length = np.concatenate((target_count_afq[:,:,0], target_count_wma[:,:,0])).reshape((-1,))
    target_nodes = np.concatenate((target_count_afq[:,:,1], target_count_wma[:,:,1])).reshape((-1,))
    target_avg_len = np.concatenate((target_count_afq[:,:,2], target_count_wma[:,:,2])).reshape((-1,))


    ### PLOT ###
    
    plt.interactive(True)
    color_list = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y']
    markers_list = ['o','o', '^','^', '*', '*', 'd','d']


    #PLOT DSC vs target length
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_length, DSC_vect)
    print("The R value is %s" %r_value)
   
    plt.figure()
    plt.subplot(311)
    for j in range(len(tract_name_list)/2):
        plt.scatter(target_count_afq[:,j,0], results_afq_m[:,j,0], c='k', marker=markers_list[j], s=40)
        plt.scatter(target_count_wma[:,j,0], results_wma_m[:,j,0], c='y', marker=markers_list[j], s=40)
    plt.plot(target_length, intercept + slope*target_length, c='r', linestyle=':')
    plt.xlabel("number of streamlines target tract", fontsize=12)
    plt.ylabel("DSC", fontsize=16)
    plt.xlim([0, np.max(target_length)+100])
    plt.ylim([0.4,0.9])
    plt.title('Mean DSC across %s subjects: R = %0.3f' %(len(example_list),r_value), fontsize=16)
    plt.legend(loc=4, fontsize='small')

    #PLOT DSC vs number nodes
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_nodes, DSC_vect)
    print("The R value is %s" %r_value)

    plt.subplot(312)	
    for j in range(len(tract_name_list)/2):
        plt.scatter(target_count_afq[:,j,1], results_afq_m[:,j,0], c='k', marker=markers_list[j], s=40)
        plt.scatter(target_count_wma[:,j,1], results_wma_m[:,j,0], c='y', marker=markers_list[j], s=40)
    plt.plot(target_nodes, intercept + slope*target_nodes, c='r', linestyle=':')
    plt.xlabel("number of nodes target tract", fontsize=12)
    plt.ylabel("DSC", fontsize=16)
    plt.xlim([0, np.max(target_nodes)+100])
    plt.ylim([0.4,0.9])
    plt.title('Mean DSC across %s subjects: R = %0.3f' %(len(example_list),r_value), fontsize=16)
    plt.legend(loc=4, fontsize='small')

    #PLOT DSC vs average streamlines length
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_avg_len, DSC_vect)
    print("The R value is %s" %r_value)
   
    plt.subplot(313)	
    for j in range(len(tract_name_list)/2):
        plt.scatter(target_count_afq[:,j,2], results_afq_m[:,j,0], c='k', marker=markers_list[j], s=40)
        plt.scatter(target_count_wma[:,j,2], results_wma_m[:,j,0], c='y', marker=markers_list[j], s=40)
    plt.plot(target_avg_len, intercept + slope*target_avg_len, c='r', linestyle=':')
    plt.xlabel("average length target tract", fontsize=12)
    plt.ylabel("DSC", fontsize=16)
    plt.xlim([0, np.max(target_avg_len)+50])
    plt.ylim([0.4,0.9])
    plt.title('Mean DSC across %s subjects: R = %0.3f' %(len(example_list),r_value), fontsize=16)
    plt.legend(loc=4, fontsize='small')


    #PLOT wDSC vs target length
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_length, wDSC_vect)
    print("The R value is %s" %r_value)
   
    plt.figure()
    plt.subplot(311)	
    plt.scatter(target_count_afq[:,:,0], wDSC_afq_vect, c='k', marker=markers_list[0], s=40)
    plt.scatter(target_count_wma[:,:,0], wDSC_wma_vect, c='y', marker=markers_list[0], s=40)
    plt.plot(target_length, intercept + slope*target_length, c='r', linestyle=':')
    plt.xlabel("number of streamlines target tract", fontsize=12)
    plt.ylabel("wDSC", fontsize=16)
    plt.xlim([0, np.max(target_length)+100])
    plt.ylim([0,1])
    plt.title('Mean wDSC across %s subjects: R = %0.3f' %(len(example_list),r_value), fontsize=16)
    plt.legend(loc=4, fontsize='small')

    #PLOT wDSC vs number nodes
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_nodes, wDSC_vect)
    print("The R value is %s" %r_value)

    plt.subplot(312)	
    plt.scatter(target_count_afq[:,:,1], wDSC_afq_vect, c='k', marker=markers_list[0], s=40)
    plt.scatter(target_count_wma[:,:,1], wDSC_wma_vect, c='y', marker=markers_list[0], s=40)
    plt.plot(target_nodes, intercept + slope*target_nodes, c='r', linestyle=':')
    plt.xlabel("number of nodes target tract", fontsize=12)
    plt.ylabel("wDSC", fontsize=16)
    plt.xlim([0, np.max(target_nodes)+100])
    plt.ylim([0,1])
    plt.title('Mean wDSC across %s subjects: R = %0.3f' %(len(example_list),r_value), fontsize=16)
    plt.legend(loc=4, fontsize='small')

    #PLOT wDSC vs average streamlines length
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(target_avg_len, wDSC_vect)
    print("The R value is %s" %r_value)
   
    plt.subplot(313)	
    plt.scatter(target_count_afq[:,:,2], wDSC_afq_vect, c='k', marker=markers_list[0], s=40)
    plt.scatter(target_count_wma[:,:,2], wDSC_wma_vect, c='y', marker=markers_list[0], s=40)
    plt.plot(target_avg_len, intercept + slope*target_avg_len, c='r', linestyle=':')
    plt.xlabel("average length target tract", fontsize=12)
    plt.ylabel("wDSC", fontsize=16)
    plt.xlim([0, np.max(target_avg_len)+50])
    plt.ylim([0,1])
    plt.title('Mean wDSC across %s subjects: R = %0.3f' %(len(example_list),r_value), fontsize=16)
    plt.legend(loc=4, fontsize='small')

    plt.show()
