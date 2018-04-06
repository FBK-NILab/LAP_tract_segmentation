"""Experiment 3.
Run LAP outside BL for partitions in B.
"""

from __future__ import print_function
import nibabel as nib
import numpy as np
from lap_multiple_examples import lap_multiple_examples, ranking_schema
import matplotlib.pyplot as plt


if __name__ == '__main__':

    experiment = 'exp3' 
    sub_list = ['990366', '991267', '993675', '996782', '992673', '992774', '995174', '983773', '910241', '910443', '911849', '912447', '917255', '917558', '919966']
    tract_name_list = ['Left_IFOF', 'Left_ILF'] #['Left_Arcuate', 'Callosum_Forceps_Minor'] # 'Right_Cingulum_Cingulate', 'Callosum_Forceps_Major']
    partition_list = ['B4', 'B8', 'B12', 'B16'] 
    src_dir = '/N/dc2/projects/lifebid/giulia/data'
    results_dir = '/N/dc2/projects/lifebid/giulia/results/%s' %experiment

    for s, sub in enumerate(sub_list):
	
	static_tractogram = '%s/HCP3_processed_data_trk/%s/%s_output_fe.trk' %(src_dir, sub, sub)

    	for t, tract_name in enumerate(tract_name_list):

    	    for p, partition in enumerate(partition_list):
	
	    	moving_tractograms_dir = '%s/partition%s/tractograms_dir' %(src_dir, partition)
		ex_dir = '%s/partition%s/examples_%s_dir' %(src_dir, partition, tract_name)
		out_filename = '%s/%s/%s_%s_tract_%s.tck' %(results_dir, sub, sub, tract_name, partition)	

		result_lap = lap_multiple_examples(moving_tractograms_dir, static_tractogram, ex_dir, out_filename)
		
		np_lap_filename = '%s/%s/%s_%s_result_lap_%s' %(results_dir, sub, sub, tract_name, partition)
		np.save(np_lap_filename, result_lap)
		
		idx_ranked = np.load('estimated_bundle_idx_ranked.npy')
		np_rank_filename = '%s/%s/%s_%s_idx_ranked_%s' %(results_dir, sub, sub, tract_name, partition)
		np.save(np_rank_filename, idx_ranked)


