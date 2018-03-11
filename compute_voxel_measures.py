"""Compute DSC"""

from __future__ import print_function
import nibabel as nib
import numpy as np
import dipy
from nibabel.streamlines import load, save 
from dipy.tracking.utils import affine_for_trackvis
from dipy.tracking.vox2track import streamline_mapping
from dipy.tracking.distances import bundles_distances_mam
from dipy.tracking.life import voxel2streamline
from dipy.viz import fvtk
from dissimilarity import compute_dissimilarity, dissimilarity
from sklearn.neighbors import KDTree
from time import sleep
import matplotlib.pyplot as plt


def compute_kdtree_and_dr_tractogram(tractogram, num_prototypes=None):
    """Compute the dissimilarity representation of the target tractogram and 
    build the kd-tree.
    """
    tractogram = np.array(tractogram, dtype=np.object)
    print("Computing dissimilarity matrices...")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012."
              % num_prototypes)
    print("Using %s prototypes" % num_prototypes)
    dm_tractogram, prototype_idx = compute_dissimilarity(tractogram,
                                                         num_prototypes=num_prototypes,
                                                         distance=bundles_distances_mam,
                                                         prototype_policy='sff',
                                                         n_jobs=-1,
                                                         verbose=False)
    prototypes = tractogram[prototype_idx]
    print("Building the KD-tree of tractogram.")
    kdt = KDTree(dm_tractogram)
    return kdt, prototypes   


def streamlines_idx(target_tract, kdt, prototypes, distance_func, warning_threshold=1.0e-4):
    """Retrieve indexes of the streamlines of the target tract.
    """
    dm_target_tract = distance_func(target_tract, prototypes)
    D, I = kdt.query(dm_target_tract, k=1)
    # print("D: %s" % D)
    # print("I: %s" % I)
    # assert((D < 1.0e-4).all())
    if (D > warning_threshold).any():
        print("WARNING (streamlines_idx()): for %s streamlines D > 1.0e-4 !!" % (D > warning_threshold).sum())
    print(D)
    target_tract_idx = I.squeeze()
    return target_tract_idx 


def compute_voxel_measures(estimated_tract_filename, true_tract_filename):

    #Loading tracts
    estimated_tract = nib.streamlines.load(estimated_tract_filename)
    true_tract = nib.streamlines.load(true_tract_filename)
    affine_est = estimated_tract.affine
    affine_true = true_tract.affine
    estimated_tract = estimated_tract.streamlines
    true_tract = true_tract.streamlines

    #affine_true=affine_for_trackvis([1.25, 1.25, 1.25])
    aff=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0,0,1.25,-72],[0,0,0,1]])

    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()
    #voxel_list_estimated_tract = dipy.tracking.life.voxel2streamline(estimated_tract, affine=affine_est)
    #voxel_list_true_tract = dipy.tracking.life.voxel2streamline(true_tract, affine=affine_true)
    #print(voxel_list_estimated_tract)
    #print(voxel_list_true_tract)
    
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)

    return DSC, TP, vol_A, vol_B


if __name__ == '__main__':

    #estimated_tract_filename = '../initial_tests/0007_Left_Cingulum_Cingulate_tract.trk'
    #estimated_tract_filename = '../initial_tests/0007_cbL_tract_IU5.trk'
    #true_tract_filename = '../initial_tests/0007_Left_Cingulum_Cingulate_tract_true.trk'

    sub = '100610'
    tract_name = 'Left_Arcuate'
    experiment = 'test' #'exp2'
    examples_list = ['0005', '0006', '0007', '0008']
    true_tracts_dir = '/N/dc2/projects/lifebid/giulia/data/HCP3_processed_data_trk'
    results_dir = '/N/dc2/projects/lifebid/giulia/data/results/%s' %experiment

    DSC_values = []
    cost_values = []

    #true_target_tract_filename = '%s/%s/%s_%s_tract.trk' %(true_tracts_dir, sub, sub, tract_name)
    #true_tract_filename = '/N/u/gberto/Karst/LAP_tract_segmentation/preliminary_tests/0008_afL_self.trk'
    true_tract_filename = '/N/u/gberto/Karst/Downloads/0007_Left_Cingulum_Cingulate_tract_E0008.tck'

    for example in examples_list:
	
	estimated_tract_filename = '%s/%s/%s_%s_tract_E%s.tck' %(results_dir, sub, sub, tract_name, example)	

    	DSC, TP, vol_A, vol_B = compute_voxel_measures(estimated_tract_filename, true_tract_filename)
    	print("The DSC value is %s" %DSC)

	result_lap = np.load('%s/%s/%s_%s_result_lap_E%s.npy' %(results_dir, sub, sub, tract_name, example))
	min_cost_values = result_lap[1]
	cost = np.sum(min_cost_values)/len(min_cost_values)

	DSC_values.append(DSC)
	cost_values.append(cost)

    #debugging
    DSC, TP, vol_A, vol_B = compute_voxel_measures(estimated_tract_filename, estimated_tract_filename)
    print("The DSC value is %s (must be 1)" %DSC)
    DSC, TP, vol_A, vol_B = compute_voxel_measures(true_tract_filename, true_tract_filename)
    print("The DSC value is %s (must be 1)" %DSC)

    # plot
    plt.figure()
    plt.scatter(DSC_values, cost_values, c="g", alpha=0.5, marker=r'$\clubsuit$', label="Luck")
    plt.xlabel("DSC")
    plt.ylabel("cost")
    plt.legend(loc=1)
    plt.show()









