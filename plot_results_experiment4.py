"""Plot the results of experiment 4.
Use the results of some experiments of exp1.
"""

from __future__ import print_function
import nibabel as nib
import numpy as np
import dipy
from nibabel.streamlines import load, save 
from compute_streamline_measures import compute_roc_curve_lap, compute_y_vectors_lap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


if __name__ == '__main__':

    experiment = 'exp1' 
    sub_list = ['983773', '990366', '991267', '993675', '996782']
    tract_name_list = ['Callosum_Forceps_Minor']#['Left_Arcuate']#, 
    partition_list =  ['A12', 'A16']
    true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment

    for s, sub in enumerate(sub_list):
       
        target_tractogram_filename = '%s/%s/%s_output_fe.trk' %(true_tracts_dir, sub, sub)
        target_tractogram = nib.streamlines.load(target_tractogram_filename)
        target_tractogram = target_tractogram.streamlines

    	for t, tract_name in enumerate(tract_name_list):
	
  	    true_tract_filename = '%s/%s/%s_%s_tract.trk' %(true_tracts_dir, sub, sub, tract_name)
	    true_tract = nib.streamlines.load(true_tract_filename)
	    true_tract = true_tract.streamlines

    	    for p, partition in enumerate(partition_list):
	
		candidate_idx_ranked = np.load('%s/%s/%s_%s_idx_ranked_%s.npy' %(results_dir, sub, sub, tract_name, partition))

   		## ROC analysis

    		#result_lap = np.load('%s/%s/%s_%s_result_lap_%s.npy' %(results_dir, sub, sub, tract_name, partition))
    		#estimated_tract_idx = result_lap[0]
    		#min_cost_values = result_lap[1]
    		#estimated_tract_idx_ranked = np.argsort(min_cost_values)
    		#y_true, y_score = compute_y_vectors_lap(estimated_tract_idx, estimated_tract_idx_ranked, true_tract, target_tractogram)
		#fpr, tpr, thresholds = roc_curve(y_true, y_score)
    		#roc_auc = auc(fpr, tpr)

	        fpr, tpr, AUC = compute_roc_curve_lap(candidate_idx_ranked, true_tract, target_tractogram)

    		# Plot of the ROC curve
		plt.interactive(True)
    		plt.figure()
    		lw = 1
    		plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' %AUC)
   		plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
   		plt.xlim([0.0, 1.0])
   		plt.ylim([0.0, 1.05])
  		plt.xlabel('False Positive Rate')
  		plt.ylabel('True Positive Rate')
  		plt.title('ROC curve of estimated %s tract in sub %s from partition %s' %(tract_name, sub, partition))
   		plt.legend(loc="lower right")
   		plt.show()









