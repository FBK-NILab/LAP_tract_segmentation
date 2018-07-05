from __future__ import division
import nibabel as nib
from nibabel.streamlines import load
from dipy.tracking.metrics import winding
import numpy as np
from dipy.tracking.streamline import length, set_number_of_points
import math
from dipy.tracking.distances import bundles_distances_mam
import time
from dipy.tracking import metrics as tm


def compute_angle_vector(str1,str2):
    angle_vector=np.zeros(len(str1)-1)
    for i in range(len(angle_vector)):
        seg_a=str1[i:i+2]
        seg_b=str2[i:i+2]
        angle_vector[i]=compute_angle(seg_a,seg_b)
    return angle_vector    

def compute_angle(seg_a,seg_b):
    offset=seg_a[0]-seg_b[0]
    c=np.concatenate((seg_a,seg_b+offset), axis=0)
    c=c[1:len(c)]
    angle=winding(c)
    if (abs(angle-360)<10e-4 or math.isnan(angle)):
        angle = 0.0
    elif (angle>180):
        angle = angle - 180          
    rad=np.deg2rad(angle)
    return rad

def compute_shape_similarity_new(str1,str2):
    step_size1 = length(str1)/len(str1)
    step_size2 = length(str2)/len(str2)
    alpha = compute_angle_vector(str1,str2)
    sim_num = (step_size1+step_size2)*np.sum(alpha)
    sim_den=math.pi*(length(str1)+length(str2))    
    sim=1-(sim_num/sim_den)  
    return sim

def compute_shape_similarity(str1,str2):
    sim_num=0
    for i in range(len(str1)-1):
        seg_a=str1[i:i+2]
        seg_b=str2[i:i+2]
        num=(length(seg_a)+length(seg_b))*compute_angle(seg_a,seg_b)
        sim_num=sim_num+num
    sim_den=math.pi*(length(str1)+length(str2))    
    sim=1-(sim_num/sim_den)  
    return sim    

def compute_bundle_shape_similarity_new(streamlines_A, streamlines_B):
    result = np.zeros((len(streamlines_A), len(streamlines_B)))
    for i, sa in enumerate(streamlines_A):
        for j, sb in enumerate(streamlines_B):
            result[i, j] = compute_shape_similarity_new(sa, sb)
    return result

def compute_bundle_shape_similarity(streamlines_A, streamlines_B):
    result = np.zeros((len(streamlines_A), len(streamlines_B)))
    for i, sa in enumerate(streamlines_A):
        for j, sb in enumerate(streamlines_B):
            result[i, j] = compute_shape_similarity(sa, sb)
    return result    

def compute_bundle_shape_similarity_flip(streamlines_A, streamlines_B):
    result1 = np.zeros((len(streamlines_A), len(streamlines_B)))
    result2 = np.zeros((len(streamlines_A), len(streamlines_B)))
    for i, sa in enumerate(streamlines_A):
        for j, sb in enumerate(streamlines_B):
            result1[i, j] = compute_shape_similarity_new(sa, sb)
	    result2[i, j] = compute_shape_similarity_new(sa, np.flip(sb, axis=0))
	print("Row %s done." %i)
    return np.fmax(result1, result2) 

def winding_diff(streamlines_A, streamlines_B):
    result = np.zeros((len(streamlines_A), len(streamlines_B)))
    for i, sa in enumerate(streamlines_A):
	for j, sb in enumerate(streamlines_B):
	    result[i, j] = abs(winding(sa)-winding(sb))
	print("Row %s done." %i)
    return result

def curvature_diff(streamlines_A, streamlines_B):
    result = np.zeros((len(streamlines_A), len(streamlines_B)))
    for i, sa in enumerate(streamlines_A):
        for j, sb in enumerate(streamlines_B):
            result[i, j] = abs(mean_curvature(sa)-mean_curvature(sb))
	print("Row %s done." %i)
    return result

def frenet_diff(streamlines_A, streamlines_B, nbp=12):
    streamlines_A_res = np.array([set_number_of_points(s, nb_points=nbp)
                               for s in streamlines_A])
    streamlines_B_res = np.array([set_number_of_points(s, nb_points=nbp)
                               for s in streamlines_B])
    result = np.zeros((len(streamlines_A_res), len(streamlines_B_res)))
    for i, sa in enumerate(streamlines_A_res):
	Ta,Na,Ba,ka,ta=tm.frenet_serret(sa)
        for j, sb in enumerate(streamlines_B_res):
	    Tb,Nb,Bb,kb,tb=tm.frenet_serret(sb)
	    #m = np.sum(np.square(ka-kb))
	    #n = np.sum(np.square(ka-np.flip(kb,axis=0)))
	    m = np.mean(abs(ka-kb))
	    n = np.mean(abs(ka-np.flip(kb,axis=0)))
            result[i, j] = min(m,n)
	#print("Row %s done." %i)
    return result


if __name__ == '__main__':

    # Load data
    #ioff_left_trk = nib.streamlines.load('/home/giulia/Desktop/test_shape/sub-660951_var-FNALW_ioff.left.trk')
    #ioff_right_trk = nib.streamlines.load('/home/giulia/Desktop/test_shape/sub-660951_var-FNALW_ioff.right.trk')
    #ioff_left_trk = nib.streamlines.load('/home/giulia/prni2017_streamline_distances/code/data/100307/wmql_tracts/100307_ifof.left.trk')
    #ioff_right_trk = nib.streamlines.load('/home/giulia/prni2017_streamline_distances/code/data/100307/wmql_tracts/100307_ifof.right.trk')
    cbL = nib.streamlines.load('/N/u/gberto/Karst/LAP_tract_segmentation/0007_Left_Cingulum_Cingulate_tract_IU5.trk')
    cbL_true = nib.streamlines.load('/N/u/gberto/Karst/LAP_tract_segmentation/Left_Cingulum_Cingulate_tract.trk')


    # Loading with trackvis.read
    #ioff_left_trk, hdr = nib.trackvis.read('/home/giulia/Desktop/test_shape/sub-660951_var-FNALW_ioff.left.trk')
    #ioff_left_trk = np.array([streamline[0] for streamline in ioff_left_trk], dtype=np.object)
    #ioff_right_trk, hdr = nib.trackvis.read('/home/giulia/Desktop/test_shape/sub-660951_var-FNALW_ioff.right.trk')
    #ioff_right_trk = np.array([streamline[0] for streamline in ioff_right_trk], dtype=np.object)

    np.set_printoptions(precision=3)

    # Extract 2 streamlines
    #str1 = ioff_left_trk.streamlines[0] 
    #str1 = ioff_left_trk[0]
    str1 = cbL.streamlines[0]
    print("Length first streamline: %s" %len(str1)) 
    #str2 = ioff_left_trk.streamlines[1] 
    #str2 = ioff_left_trk[1]
    str2 = cbL.streamlines[0]
    print("Length second streamline: %s" %len(str2)) 

    # Resample streamlines
    nbp=12
    str1_res = set_number_of_points(str1, nb_points=nbp)
    str2_res = set_number_of_points(str2, nb_points=nbp)
    print("Length first streamline after resample: %s" %len(str1_res)) 
    print("Length second streamline after resample: %s" %len(str2_res)) 

    # Compute similarity measure
    similarity_coeff = compute_shape_similarity(str1_res, str2_res)
    print("Similarity coefficient: %s" %similarity_coeff)

    # Compute bundle similarity matrix 
    #streamlines_A = ioff_left_trk.streamlines[0:100]
    #streamlines_B = ioff_right_trk.streamlines[0:150]
    #streamlines_A = ioff_left_trk[0:100]
    #streamlines_B = ioff_right_trk[0:150]
    streamlines_A = cbL.streamlines[0:200]
    streamlines_B = cbL_true.streamlines[0:300]
    streamlines_A_res = np.array([set_number_of_points(s, nb_points=nbp)
                               for s in streamlines_A])
    streamlines_B_res = np.array([set_number_of_points(s, nb_points=nbp)
                               for s in streamlines_B])
    t0=time.time()
    similarity_bundle_matrix = compute_bundle_shape_similarity(streamlines_A_res, streamlines_B_res)
    t1=time.time()
    #print("Similarity bundle matrix: %s" %similarity_bundle_matrix)

    #Compute distance matrix
    t2=time.time()
    D = bundles_distances_mam(streamlines_A, streamlines_B)
    t3=time.time()
    #print("Distance matrix: %s" %D)

    t4=time.time()
    similarity_bundle_matrix_new = compute_bundle_shape_similarity_new(streamlines_A_res, streamlines_B_res)
    t5=time.time()

    print("Time to compute the shape matrix = %s seconds" %(t1-t0))
    print("Time to compute the distance matrix = %s seconds" %(t3-t2))
    print("Time to compute the shape matrix new = %s seconds" %(t5-t4))
