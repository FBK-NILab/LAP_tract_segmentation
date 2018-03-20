"""Compute some streamline measures"""

from __future__ import print_function, division
import numpy as np
import nibabel as nib
from nibabel.streamlines import load
from dipy.tracking.distances import bundles_distances_mam
from lap_single_example import compute_kdtree_and_dr_tractogram
from dissimilarity import compute_dissimilarity, dissimilarity
from sklearn.neighbors import KDTree


def compute_loss_function(source_tract, ett):
    """Compute the loss function between two tracts. 
    """
    sP = len(source_tract) 
    sQ = len(ett)
    distance_matrix = bundles_distances_mam(source_tract, ett, metric='avg') 
    L = np.sum(distance_matrix)
    L = L / (sP*sQ)   
    return L     


def compute_bmd(source_tract, ett):
    """Compute the cost function Bundle-based Minimum Distance (BMD) 
    as in [Garyfallidis et al. 2015], but using the mam_avg distance 
    instead of the MDF distance. 
    """
    A = len(source_tract) 
    B = len(ett)
    distance_matrix = bundles_distances_mam(source_tract, ett, metric='avg')

    min_a = 0
    min_b = 0
    for j in range(A):
        min_a = min_a + np.min(distance_matrix[j])
    for i in range(B):
        min_b = min_b + np.min(distance_matrix[:,i]) 
    BMD = ((min_a/A + min_b/B)**2)/4    

    return BMD 


def compute_loss_and_bmd(source_tract, ett):
    """Compute loss function and BMD.
    """
    A = len(source_tract) 
    B = len(ett)
    distance_matrix = bundles_distances_mam(source_tract, ett, metric='avg')

    L = np.sum(distance_matrix)
    L = L / (A*B)

    min_a = 0
    min_b = 0
    for j in range(A):
        min_a = min_a + np.min(distance_matrix[j])
    for i in range(B):
        min_b = min_b + np.min(distance_matrix[:,i]) 
    BMD = ((min_a/A + min_b/B)**2)/4    

    return L, BMD 


def compute_superset(true_tract, kdt, prototypes, k=1000, distance_func=bundles_distances_mam):
    """Compute a superset of the true target tract with k-NN.
    """
    true_tract = np.array(true_tract, dtype=np.object)
    dm_true_tract = distance_func(true_tract, prototypes)
    D, I = kdt.query(dm_true_tract, k=k)
    superset_idx = np.unique(I.flat)
    return superset_idx


def streamlines_idx(target_tract, kdt, prototypes, distance_func=bundles_distances_mam, warning_threshold=1.0e-4):
    """Retrieve indexes of the streamlines of the target tract.
    """
    dm_target_tract = distance_func(target_tract, prototypes)
    D, I = kdt.query(dm_target_tract, k=1)
    if (D > warning_threshold).any():
        print("WARNING (streamlines_idx()): for %s streamlines D > 1.0e-4 !!" % (D > warning_threshold).sum())
    print(D)
    target_tract_idx = I.squeeze()
    return target_tract_idx 


def compute_roc_curve_lap(result_lap, true_tract, target_tractogram):
    """Compute ROC curve.
    """ 
    print("Compute the dissimilarity representation of the target tractogram and build the kd-tree.")
    kdt, prototypes = compute_kdtree_and_dr_tractogram(target_tractogram)

    print("Compute a superset of the true target tract with k-NN.")
    superset_idx = compute_superset(true_tract, kdt, prototypes)

    print("Retrieving indeces of the true_tract")
    true_tract_idx = streamlines_idx(true_tract, kdt, prototypes)

    print("Computing FPR and TPR.")
    y_true = np.zeros(len(superset_idx))
    correspondent_idx = np.array([np.where(superset_idx==true_tract_idx[i]) for i in range(len(true_tract_idx))])
    y_true[correspondent_idx] = 1

    min_cost_values = result_lap[1]
    estimated_tract_idx = result_lap[0]
    m = np.argsort(min_cost_values)

    c=len(m)
    y_score = c*np.ones(len(superset_idx))

    f=np.array([np.where(superset_idx==estimated_tract_idx[i]) for i in range(len(estimated_tract_idx))])
    h=f	

    for i in range(c):
        y_score[h[i]] = m[i]

    y_score=abs(y_score-c)

    return superset_idx, true_tract_idx, correspondent_idx, y_true, y_score, f, h, m


def compute_y_vectors_lap(estimated_tract_idx, estimated_tract_idx_ranked, true_tract, target_tractogram):
    """Compute y_true and y_score.
    """ 
    print("Compute the dissimilarity representation of the target tractogram and build the kd-tree.")
    kdt, prototypes = compute_kdtree_and_dr_tractogram(target_tractogram)

    print("Compute a superset of the true target tract with k-NN.")
    superset_idx = compute_superset(true_tract, kdt, prototypes)

    print("Retrieving indeces of the true_tract")
    true_tract_idx = streamlines_idx(true_tract, kdt, prototypes)

    print("Compute y_true.")
    y_true = np.zeros(len(superset_idx))
    correspondent_idx_true = np.array([np.where(superset_idx==true_tract_idx[i]) for i in range(len(true_tract_idx))])
    y_true[correspondent_idx_true] = 1

    print("Compute y_score.")
    S = len(estimated_tract_idx)
    y_score = S*np.ones(len(superset_idx))
    correspondent_idx_score = np.array([np.where(superset_idx==estimated_tract_idx[i]) for i in range(S)])
    for i in range(S):
        y_score[correspondent_idx_score[i]] = estimated_tract_idx_ranked[i]
    #invert the ranking   
    y_score = abs(y_score-S)

    return superset_idx, true_tract_idx, correspondent_idx_true, correspondent_idx_score, y_true, y_score













