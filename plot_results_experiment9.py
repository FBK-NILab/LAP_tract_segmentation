"""Plot the results of experiment 9 that uses the results of experiment2 (LAP single example)."""

from __future__ import print_function, division
import nibabel as nib
import numpy as np
import dipy
import scipy.stats
from nibabel.streamlines import load, save 
from compute_voxel_measures import compute_voxel_measures
import matplotlib.pyplot as plt
from dipy.tracking.streamline import length, set_number_of_points
from dipy.tracking import metrics as tm


def mean_curvature(streamlines_A, nbp=256):
    streamlines_A_res = np.array([set_number_of_points(s, nb_points=nbp)
                               for s in streamlines_A])
    curvature_A = []
    for sa in streamlines_A_res:
	Ta,Na,Ba,ka,ta=tm.frenet_serret(sa)
        curvature_A.append(ka)
    return np.mean(curvature_A)


if __name__ == '__main__':

    experiment = 'exp2'
    sub_list = ['993675', '995174', '996782', '992673', '992774']
    tract_name_list = ['Left_Arcuate', 'Callosum_Forceps_Minor', 'Left_IFOF', 'Left_ILF', 'Left_Uncinate', 'Left_Cingulum_Hippocampus']
    example_list = ['615441', '615744', '616645', '617748', '618952', '633847', '634748', '635245', '638049']
    true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    examples_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment

    DSC_values = np.zeros((len(sub_list), len(tract_name_list), len(example_list)))
    wDSC_values = np.zeros((len(sub_list), len(tract_name_list), len(example_list)))
    J_values = np.zeros((len(sub_list), len(tract_name_list), len(example_list)))
    sensitivity_values = np.zeros((len(sub_list), len(tract_name_list), len(example_list)))
    len_target_values = np.zeros((len(sub_list), len(tract_name_list)))
    ml_target_values = np.zeros((len(sub_list), len(tract_name_list)))
    mc_target_values = np.zeros((len(sub_list), len(tract_name_list)))
    len_example_values = np.zeros((len(example_list), len(tract_name_list)))
    len_ratio_values = np.zeros((len(sub_list), len(tract_name_list), len(example_list)))

    for s, sub in enumerate(sub_list):

    	for t, tract_name in enumerate(tract_name_list):
	
  	    true_tract_filename = '%s/%s/%s_%s_tract.trk' %(true_tracts_dir, sub, sub, tract_name)
	    true_tract = nib.streamlines.load(true_tract_filename)
	    true_tract = true_tract.streamlines
	    len_target = len(true_tract)
	    ml_target = np.mean(length(true_tract))
	    mc_target = mean_curvature(true_tract)
	    len_target_values[s,t] = len_target
	    ml_target_values[s,t] = ml_target
	    mc_target_values[s,t] = mc_target


    	    for e, example in enumerate(example_list):

		example_tract_filename = '%s/%s/%s_%s_tract.trk' %(examples_dir, example, example, tract_name)
		example_tract = nib.streamlines.load(example_tract_filename)
		example_tract = example_tract.streamlines
		len_example = len(example_tract)
		len_example_values[e,t] = len_example
	
	    	estimated_tract_filename = '%s/%s/%s_%s_tract_E%s.tck' %(results_dir, sub, sub, tract_name, example)	
		estimated_tract = nib.streamlines.load(estimated_tract_filename)
	        estimated_tract = estimated_tract.streamlines

	    	DSC, wDSC, J, sensitivity, vol_A, vol_B  = compute_voxel_measures(estimated_tract, true_tract)	
	    	print("The DSC value is %s" %DSC)
	
	    	DSC_values[s,t,e] = DSC
	    	wDSC_values[s,t,e] = wDSC
		J_values[s,t,e] = J
		sensitivity_values[s,t,e] = sensitivity
		len_ratio_values[s,t,e] =  len_example / len_target 


    #reshape arrays
    DSC_vect_m = np.mean(DSC_values, axis=2).reshape((-1,))
    wDSC_vect_m = np.mean(wDSC_values, axis=2).reshape((-1,))
    len_target_vect = len_target_values.reshape((-1,))
    len_example_vect = len_example_values.reshape((-1,))
    ml_target_vect = ml_target_values.reshape((-1,))
    mc_target_vect = mc_target_values.reshape((-1,))

    DSC_vect = DSC_values.reshape((-1,))
    wDSC_vect = wDSC_values.reshape((-1,))
    len_ratio_vect = len_ratio_values.reshape((-1,))


    ### PLOT ###
    
    plt.interactive(True)
    color_list = ['k', 'g', 'r', 'y', 'b', 'c']
    markers_list = ['o', '^', '*', 'd',  'p', '+']


    #PLOT DSC vs target length
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(len_target_vect, DSC_vect_m)
    print("The R value is %s" %r_value)
   
    plt.figure()
    plt.subplot(211)
    for j in range(len(tract_name_list)):
	plt.scatter(len_target_values[:,j], np.mean(DSC_values[:,j,:], axis=1), c=color_list[0],  marker=markers_list[j], s=70, label=tract_name_list[j])	
    plt.plot(len_target_vect, intercept + slope*len_target_vect, c='r', linestyle=':')
    plt.xlabel("size target tract", fontsize=16)
    plt.ylabel("DSC", fontsize=16)
    plt.xlim([0, np.max(len_target_vect)+100])
    plt.ylim([0,1])
    plt.title('Mean DSC across %s subjects: R = %0.3f' %(len(example_list),r_value), fontsize=20)
    plt.legend(loc=4, fontsize='small')

    #PLOT wDSC vs target length
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(len_target_vect, wDSC_vect_m)
    print("The R value is %s" %r_value)
   
    plt.subplot(212)
    #for i in range(len(sub_list)):
    for j in range(len(tract_name_list)):
	plt.scatter(len_target_values[:,j], np.mean(wDSC_values[:,j,:], axis=1), c=color_list[0],  marker=markers_list[j], s=70, label=tract_name_list[j])	
    plt.plot(len_target_vect, intercept + slope*len_target_vect, c='r', linestyle=':')
    plt.xlabel("size target tract", fontsize=16)
    plt.ylabel("wDSC", fontsize=16)
    plt.xlim([0, np.max(len_target_vect)+100])
    plt.ylim([0,1])
    plt.title('Mean wDSC across %s subjects: R = %0.3f' %(len(example_list),r_value), fontsize=20)
    plt.legend(loc=4, fontsize='small')
    


    #PLOT DSC vs target ml
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ml_target_vect, DSC_vect_m)
    print("The R value is %s" %r_value)
   
    plt.figure()
    plt.subplot(211)
    for j in range(len(tract_name_list)):
	plt.scatter(ml_target_values[:,j], np.mean(DSC_values[:,j,:], axis=1), c=color_list[0],  marker=markers_list[j], s=70, label=tract_name_list[j])	
    plt.plot(ml_target_vect, intercept + slope*ml_target_vect, c='r', linestyle=':')
    plt.xlabel("mean length target tract")
    plt.ylabel("DSC")
    plt.xlim([0, np.max(ml_target_vect)+20])
    plt.ylim([0,1])
    plt.title('Mean DSC across %s subjects: R = %0.3f' %(len(example_list),r_value))
    plt.legend(loc=4, fontsize='small')

    #PLOT wDSC vs target ml
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(ml_target_vect, wDSC_vect_m)
    print("The R value is %s" %r_value)
   
    plt.subplot(212)
    #for i in range(len(sub_list)):
    for j in range(len(tract_name_list)):
	plt.scatter(ml_target_values[:,j], np.mean(wDSC_values[:,j,:], axis=1), c=color_list[0],  marker=markers_list[j], s=70, label=tract_name_list[j])	
    plt.plot(ml_target_vect, intercept + slope*ml_target_vect, c='r', linestyle=':')
    plt.xlabel("mean length target tract")
    plt.ylabel("wDSC")
    plt.xlim([0, np.max(ml_target_vect)+20])
    plt.ylim([0,1])
    plt.title('Mean wDSC across %s subjects: R = %0.3f' %(len(example_list),r_value))
    plt.legend(loc=4, fontsize='small')



    #PLOT DSC vs target mc
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mc_target_vect, DSC_vect_m)
    print("The R value is %s" %r_value)
   
    plt.figure()
    plt.subplot(211)
    for j in range(len(tract_name_list)):
	plt.scatter(mc_target_values[:,j], np.mean(DSC_values[:,j,:], axis=1), c=color_list[0],  marker=markers_list[j], s=70, label=tract_name_list[j])	
    plt.plot(mc_target_vect, intercept + slope*mc_target_vect, c='r', linestyle=':')
    plt.xlabel("mean curvature target tract")
    plt.ylabel("DSC")
    plt.xlim([0, np.max(mc_target_vect)])
    plt.ylim([0,1])
    plt.title('Mean DSC across %s subjects: R = %0.3f' %(len(example_list),r_value))
    plt.legend(loc=0, fontsize='small')

    #PLOT wDSC vs target mc
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mc_target_vect, wDSC_vect_m)
    print("The R value is %s" %r_value)
   
    plt.subplot(212)
    #for i in range(len(sub_list)):
    for j in range(len(tract_name_list)):
	plt.scatter(mc_target_values[:,j], np.mean(wDSC_values[:,j,:], axis=1), c=color_list[0],  marker=markers_list[j], s=70, label=tract_name_list[j])	
    plt.plot(mc_target_vect, intercept + slope*mc_target_vect, c='r', linestyle=':')
    plt.xlabel("mean curvature target tract")
    plt.ylabel("wDSC")
    plt.xlim([0, np.max(mc_target_vect)])
    plt.ylim([0,1])
    plt.title('Mean wDSC across %s subjects: R = %0.3f' %(len(example_list),r_value))
    plt.legend(loc=0, fontsize='small')






















    #PLOT DSC vs ratio target length / example length
    x = len_ratio_values
    #y = np.exp(-abs(x-1))
    y = -np.abs(x-1)+1
    x_vect=len_ratio_vect
    #y_vect=np.exp(-abs(x_vect-1))
    y_vect = -np.abs(x_vect-1)+1

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_vect, DSC_vect)
    print("The R value is %s" %r_value)
   
    plt.figure()
    #for i in range(len(sub_list)):
    for j in range(len(tract_name_list)):
	for k in range(len(example_list)):
	    plt.scatter(y[:,j,k], DSC_values[:,j,k], c=color_list[0],  marker=markers_list[j], s=70)	
    plt.plot(y_vect, intercept + slope*y_vect, c='r', linestyle=':')
    plt.xlabel("ratio length")
    plt.ylabel("DSC")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('Individual subjects: R = %0.3f' %r_value)
    plt.legend(loc=4, fontsize='small')



    #PLOT wDSC vs ratio target length / example length
    x = len_ratio_values
    #y = np.exp(-abs(x-1))
    y = -np.abs(x-1)+1
    x_vect=len_ratio_vect
    #y_vect=np.exp(-abs(x_vect-1))
    y_vect = -np.abs(x_vect-1)+1

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_vect, wDSC_vect)
    print("The R value is %s" %r_value)
   
    plt.figure()
    #for i in range(len(sub_list)):
    for j in range(len(tract_name_list)):
	for k in range(len(example_list)):
	    plt.scatter(y[:,j,k], wDSC_values[:,j,k], c=color_list[0],  marker=markers_list[j], s=70)	
    plt.plot(y_vect, intercept + slope*y_vect, c='r', linestyle=':')
    plt.xlabel("ratio length")
    plt.ylabel("wDSC")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('Individual subjects: R = %0.3f' %r_value)
    plt.legend(loc=4, fontsize='small')


    # removing outliers for UF
    # target ouliers: 993675(0), 992673(3)
    # examples outliers: 615744(1), 618952(4), 633847(5)
    
    new_DSC_values = DSC_values[np.ix_([1,2,4], [0,1,2,3,4,5], [0,2,3,6,7,8])]
    new_wDSC_values = wDSC_values[np.ix_([1,2,4], [0,1,2,3,4,5], [0,2,3,6,7,8])]
    new_len_target_values = len_target_values[np.ix_([1,2,4], [0,1,2,3,4,5])]

    new_DSC_vect_m = np.mean(new_DSC_values, axis=2).reshape((-1,))
    new_wDSC_vect_m = np.mean(new_wDSC_values, axis=2).reshape((-1,))
    new_len_target_vect = new_len_target_values.reshape((-1,))


    #PLOT DSC vs target length
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(new_len_target_vect, new_DSC_vect_m)
    print("The R value is %0.3f" %r_value)
   
    plt.figure()
    #for i in range(len(sub_list)):
    for j in range(len(tract_name_list)):
	plt.scatter(new_len_target_values[:,j], np.mean(new_DSC_values[:,j,:], axis=1), c=color_list[0],  marker=markers_list[j], s=70, label=tract_name_list[j])	
    plt.plot(new_len_target_vect, intercept + slope*new_len_target_vect, c='r', linestyle=':')
    plt.xlabel("size target tract")
    plt.ylabel("DSC")
    plt.xlim([0, np.max(new_len_target_vect)+100])
    plt.ylim([0,1])
    plt.title('Mean DSC across %s subjects: R = %0.3f (no UF outliers)' %(len(example_list)-3, r_value))
    plt.legend(loc=4, fontsize='small')
    

    #PLOT wDSC vs target length
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(new_len_target_vect, new_wDSC_vect_m)
    print("The R value is %s" %r_value)
   
    plt.figure()
    #for i in range(len(sub_list)):
    for j in range(len(tract_name_list)):
	plt.scatter(new_len_target_values[:,j], np.mean(new_wDSC_values[:,j,:], axis=1), c=color_list[0],  marker=markers_list[j], s=70, label=tract_name_list[j])	
    plt.plot(new_len_target_vect, intercept + slope*new_len_target_vect, c='r', linestyle=':')
    plt.xlabel("size target tract")
    plt.ylabel("wDSC")
    plt.xlim([0, np.max(new_len_target_vect)+100])
    plt.ylim([0,1])
    plt.title('Mean wDSC across %s subjects: R = %0.3f (no UF outliers)' %(len(example_list)-3, r_value))
    plt.legend(loc=4, fontsize='small')
    

    plt.show()

    
    

