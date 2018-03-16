"""Experiment 1.
Run LAP outside BL.
"""

from __future__ import print_function
import nibabel as nib
import numpy as np
from lap_multiple_examples import lap_multiple_examples, ranking_schema
import matplotlib.pyplot as plt


if __name__ == '__main__':

    experiment = 'exp1' #'test' #'exp1'
    sub_list = ['983773', '990366', '991267'] #['993675', '996782'] #
    tract_name_list = ['Callosum_Forceps_Minor'] #['Left_Arcuate'] #, 'Callosum_Forceps_Minor', 'Right_Cingulum_Cingulate', 'Callosum_Forceps_Major']
    partition_list = ['A1', 'A4', 'A8'] #, 'A12', 'A16'] 
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
		
		np_filename = '%s/%s/%s_%s_result_lap_%s' %(results_dir, sub, sub, tract_name, partition)
		np.save(np_filename, result_lap)
