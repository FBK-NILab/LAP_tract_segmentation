""" Bundle segmentation with Rectangular Linear Assignment Problem.

	See Sharmin et al., 'White Matter Tract Segmentation as Multiple 
	Linear Assignment Problems', Fronts. Neurosci., 2017.
"""

import os
import sys
import argparse
import os.path
import nibabel as nib
import numpy as np
from nibabel.streamlines import load
from slr_registration import tractograms_slr
from dissimilarity import compute_dissimilarity, dissimilarity
from dipy.tracking.distances import bundles_distances_mam
from sklearn.neighbors import KDTree
from dipy.viz import fvtk

try:
    from linear_assignment import LinearAssignment
except ImportError:
    print("WARNING: Cythonized LAPJV not available. Falling back to Python.")
    print("WARNING: See README.txt")
    from linear_assignment_numpy import LinearAssignment


def compute_kdtree_and_dr_tractogram(tractogram, num_prototypes=None):
    """Compute the dissimilarity representation of the target tractogram and 
    build the kd-tree.
    """
    tractogram = np.array(tractogram)
    print("Computing dissimilarity matrices")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012"
              % num_prototypes)
    print("Using %s prototypes" % num_prototypes)
    dm_tractogram, prototype_idx = compute_dissimilarity(tractogram,
                                                         num_prototypes=num_prototypes,
                                                         distance=bundles_distances_mam,
                                                         prototype_policy='sff',
                                                         n_jobs=-1,
                                                         verbose=False)
    prototypes = tractogram[prototype_idx]
    print("Building the KD-tree of tractogram")
    kdt = KDTree(dm_tractogram)
    return kdt, prototypes    


def RLAP(kdt, k, dm_source_tract, source_tract, tractogram, distance):
    """Code for Rectangular Linear Assignment Problem.
    """
    D, I = kdt.query(dm_source_tract, k=k)
    superset = np.unique(I.flat)
    print("Computing the cost matrix (%s x %s) for RLAP " % (len(source_tract),
                                                             len(superset)))
    cost_matrix = dissimilarity(source_tract, tractogram[superset], distance)
    print("Computing RLAP with LAPJV")
    assignment = LinearAssignment(cost_matrix).solution
    return superset[assignment]


def show_both_bundles(bundles, colors=None, show=False, fname=None):
	ren = fvtk.ren()
	ren.SetBackground(1., 1, 1)
	for (i, bundle) in enumerate(bundles):
		color = colors[i]
		lines = fvtk.streamtube(bundle, color, linewidth=0.3)
		lines.RotateX(-90)
		lines.RotateZ(90)
		fvtk.add(ren, lines)
	if show:
		fvtk.show(ren)
	if fname is not None:
		sleep(1)
		fvtk.record(ren, n_frames=1, out_path=fname,size=(3000,3000))
		fvtk.show(ren)




if __name__ == '__main__':

	# Setting the location of the repository
	script_src = os.path.basename(sys.argv[0]).strip('.py')
	script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
	src_dir = os.path.abspath(os.path.join(script_dir, 'data/HCP3/derivatives'))
	#out_dir = os.path.join(src_dir, 'source_tractogram_aligned')
	#if not os.path.exists(out_dir):
	#    os.mkdir(out_dir)   

	parser = argparse.ArgumentParser()
	parser.add_argument('-target', nargs='?', const=1, default='',
	                    help='The target subject id')
	parser.add_argument('-source', nargs='?',  const=1, default='',
	                    help='The source subject id')
	parser.add_argument('-bundle', nargs='?',  const=1, default='',
	                    help='The name of the bundle')
	                    
	args = parser.parse_args()

	print("Computing the affine slr transformation.")
	affine, srm, source_tractogram_aligned = tractograms_slr(args.target, args.source, src_dir)

	print("Applying the affine to the example bundle.")
	example_bundle_filename = '%s/wmql_FNALW/sub-%s/sub-%s_var-FNALW_%s.trk' %(src_dir, args.source, args.source, args.bundle)
	print("Example bundle filename: %s" %example_bundle_filename)
	example_bundle = nib.streamlines.load(example_bundle_filename)
	example_bundle = example_bundle.streamlines
	example_bundle_aligned = srm.transform(example_bundle)

	print("Compute the dissimilarity representation of the target tractogram and build the kd-tree.")
	target_tractogram_filename = '%s/deterministic_tracking_dipy_FNAL/sub-%s/sub-%s_var-FNAL_tract.trk' %(src_dir, args.target, args.target)  
	target_tractogram = nib.streamlines.load(target_tractogram_filename)
	target_tractogram = target_tractogram.streamlines
	kdt, prototypes = compute_kdtree_and_dr_tractogram(target_tractogram)

	print("Compute the dissimilarity of the aligned example bundle with the prototypes of target tractogram.")
	dm_example_bundle, prototype_idx = compute_dissimilarity(target_tractogram,
                                                             num_prototypes=40,
                                                             distance=bundles_distances_mam,
                                                             prototype_policy='sff',
                                                             n_jobs=-1,
                                                             verbose=False)

	print("Segmentation as Rectangular linear Assignment Problem (RLAP).")
	k = 200
	distance = bundles_distances_mam
	estimated_target_bundle = RLAP(kdt, k, dm_example_bundle, example_bundle, target_tractogram, distance)

	# Visualization
	print("Loading true target bundle...")
	true_target_bundle_filename = '%s/wmql_FNALW/sub-%s/sub-%s_var-FNALW_%s.trk' %(src_dir, args.target, args.target, args.bundle)
	print("True target bundle filename: %s" %true_target_bundle_filename)
	true_target_bundle = nib.streamlines.load(true_target_bundle_filename)
	true_target_bundle = true_target_bundle.streamlines	

	show_both_bundles([true_target_bundle, estimated_target_bundle],
                       colors=[fvtk.colors.blue, fvtk.colors.red]) 

	sys.exit()    