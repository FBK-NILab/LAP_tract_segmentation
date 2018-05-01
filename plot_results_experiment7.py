from __future__ import print_function
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

	#load the matrices
	DSC_rlap_frenet = np.load('DSC_rlap_frenet_25')
	wDSC_rlap_frenet = np.load('wDSC_rlap_frenet_25')
	J_rlap_frenet = np.load('J_rlap_frenet_25')
	sensitivity_rlap_frenet = np.load('sensitivity_rlap_frenet_25')

	#compute the mean across the subjects: returns a 2D matrix
	DSC_mean = np.mean(DSC_rlap_frenet, axis=(0,2))
	wDSC_mean = np.mean(wDSC_rlap_frenet, axis=(0,2))
	J_mean = np.mean(J_rlap_frenet, axis=(0,2))
	sensitivity_mean = np.mean(sensitivity_rlap_frenet, axis=(0,2))
	
	#compute the std across the subjects: returns a 2D matrix
	DSC_std = np.std(DSC_rlap_frenet, axis=(0,2))
	wDSC_std = np.std(wDSC_rlap_frenet, axis=(0,2))
	J_mean = np.std(J_rlap_frenet, axis=(0,2))
	sensitivity_mean = np.std(sensitivity_rlap_frenet, axis=(0,2))


	# PLOT

	from basic_units import cm, inch

	tract_name_list = ['Left_IFOF', 'Left_ILF', 'Left_Arcuate', 'Callosum_Forceps_Minor']
	h_list = [1, 0.8, 0.6, 0.4, 0.2, 0]
	ind = np.arange(len(tract_name_list))    # the x locations for the groups
	width = 0.8 / len(h_list)       # the width of the bars


	fig, ax = plt.subplots()

	p1 = ax.bar(ind, DSC_mean[:,0], width, color='r', bottom=0*cm, yerr=DSC_std[:,0])
	p2 = ax.bar(ind + width, DSC_mean[:,1], width, color='y', bottom=0*cm, yerr=DSC_std[:,1])
	p3 = ax.bar(ind + 2*width, DSC_mean[:,2], width, color='g', bottom=0*cm, yerr=DSC_std[:,2])
	p4 = ax.bar(ind + 3*width, DSC_mean[:,3], width, color='c', bottom=0*cm, yerr=DSC_std[:,3])
	p5 = ax.bar(ind + 4*width, DSC_mean[:,4], width, color='b', bottom=0*cm, yerr=DSC_std[:,4])
	p6 = ax.bar(ind + 5*width, DSC_mean[:,5], width, color='m', bottom=0*cm, yerr=DSC_std[:,5])

	ax.set_title('mean DSC across 25 pairs of subjects')
	ax.set_xticks(ind + len(h_list)*width / 2)
	ax.set_xticklabels(h_list)
	ax.set_ylim(0,1)
	ax.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('h=1', 'h=0.8', 'h=0.6', 'h=0.4', 'h=0.2', 'h=0'), loc=4)
	ax.set_ylabel('DSC')
	ax.autoscale_view()


	fig, ax = plt.subplots()
	
	p1 = ax.bar(ind, wDSC_mean[:,0], width, color='r', bottom=0*cm, yerr=wDSC_std[:,0])
	p2 = ax.bar(ind + width, wDSC_mean[:,1], width, color='y', bottom=0*cm, yerr=wDSC_std[:,1])
	p3 = ax.bar(ind + 2*width, wDSC_mean[:,2], width, color='g', bottom=0*cm, yerr=wDSC_std[:,2])
	p4 = ax.bar(ind + 3*width, wDSC_mean[:,3], width, color='c', bottom=0*cm, yerr=wDSC_std[:,3])
	p5 = ax.bar(ind + 4*width, wDSC_mean[:,4], width, color='b', bottom=0*cm, yerr=wDSC_std[:,4])
	p6 = ax.bar(ind + 5*width, wDSC_mean[:,5], width, color='m', bottom=0*cm, yerr=wDSC_std[:,5])

	ax.set_title('mean wDSC across 25 pairs of subjects')
	ax.set_xticks(ind + len(h_list)*width / 2)
	ax.set_xticklabels(h_list)
	ax.set_ylim(0,1)
	ax.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('h=1', 'h=0.8', 'h=0.6', 'h=0.4', 'h=0.2', 'h=0'), loc=4)
	ax.set_ylabel('wDSC')
	ax.autoscale_view()


	fig, ax = plt.subplots()
	
	p1 = ax.bar(ind, J_mean[:,0], width, color='r', bottom=0*cm, yerr=J_std[:,0])
	p2 = ax.bar(ind + width, J_mean[:,1], width, color='y', bottom=0*cm, yerr=J_std[:,1])
	p3 = ax.bar(ind + 2*width, J_mean[:,2], width, color='g', bottom=0*cm, yerr=J_std[:,2])
	p4 = ax.bar(ind + 3*width, J_mean[:,3], width, color='c', bottom=0*cm, yerr=J_std[:,3])
	p5 = ax.bar(ind + 4*width, J_mean[:,4], width, color='b', bottom=0*cm, yerr=J_std[:,4])
	p6 = ax.bar(ind + 5*width, J_mean[:,5], width, color='m', bottom=0*cm, yerr=J_std[:,5])

	ax.set_title('mean Jaccard index across 25 pairs of subjects')
	ax.set_xticks(ind + len(h_list)*width / 2)
	ax.set_xticklabels(h_list)
	ax.set_ylim(0,1)
	ax.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('h=1', 'h=0.8', 'h=0.6', 'h=0.4', 'h=0.2', 'h=0'), loc=4)
	ax.set_ylabel('J')
	ax.autoscale_view()

	
	fig, ax = plt.subplots()
	
	p1 = ax.bar(ind, sensitivity_mean[:,0], width, color='r', bottom=0*cm, yerr=sensitivity_std[:,0])
	p2 = ax.bar(ind + width, sensitivity_mean[:,1], width, color='y', bottom=0*cm, yerr=sensitivity_std[:,1])
	p3 = ax.bar(ind + 2*width, sensitivity_mean[:,2], width, color='g', bottom=0*cm, yerr=sensitivity_std[:,2])
	p4 = ax.bar(ind + 3*width, sensitivity_mean[:,3], width, color='c', bottom=0*cm, yerr=sensitivity_std[:,3])
	p5 = ax.bar(ind + 4*width, sensitivity_mean[:,4], width, color='b', bottom=0*cm, yerr=sensitivity_std[:,4])
	p6 = ax.bar(ind + 5*width, sensitivity_mean[:,5], width, color='m', bottom=0*cm, yerr=sensitivity_std[:,5])

	ax.set_title('mean sensitivity across 25 pairs of subjects')
	ax.set_xticks(ind + len(h_list)*width / 2)
	ax.set_xticklabels(h_list)
	ax.set_ylim(0,1)
	ax.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('h=1', 'h=0.8', 'h=0.6', 'h=0.4', 'h=0.2', 'h=0'), loc=4)
	ax.set_ylabel('sensitivity')
	ax.autoscale_view()


	plt.show()
