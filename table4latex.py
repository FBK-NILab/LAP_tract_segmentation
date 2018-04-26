import numpy as np

def print_latex_table(matrix2D, column_labels, row_labels):
	print("\\begin{table}")
	print("\\centering")
	tmp = ' | '.join('c' * (len(column_labels) + 1))
	print("\\begin{tabular}{ %s }" % tmp)
	print(' & ' + ' & '.join([col for col in column_labels]) + ' \\\\')
	print('\\hline')
	print('\\hline')
	for i, row in enumerate(matrix2D):
		print('\t ' + row_labels[i] + ' & ' + ' & '.join(["%.3f" % v for v in matrix2D[i]]) + ' \\\\')
		print('\t \\hline')	
	print("\\end{tabular}")
	print("\\caption{\small{Add caption here.}}")
	print("\\end{table}\n")


if __name__ == '__main__':

	bundle_list = ['cstL', 'cstR', 'ifofL', 'ifofR', 'thprefL', 'thprefR', 'ufL', 'ufR']
	registration = ['slr', 'ant4t1w', 'ant4fa']

	DSC_nn = np.load('DSC_nn_90.npy')
	DSC_nn_masked = np.ma.masked_equal(DSC_nn, 0) 
	DSC_mean_nn = np.mean(DSC_nn_masked, axis=(0,2))

	DSC_matrix2D_nn = np.transpose(DSC_mean_nn[:,:,0])
	print_latex_table(DSC_matrix2D_nn, bundle_list, registration)


	DSC_rlap = np.load('DSC_rlap_90.npy')
	DSC_rlap_masked = np.ma.masked_equal(DSC_rlap, 0) 
	DSC_mean_rlap = np.mean(DSC_rlap_masked, axis=(0,2))

	DSC_matrix2D_rlap = np.transpose(DSC_mean_rlap[:,:,0])
	print_latex_table(DSC_matrix2D_rlap, bundle_list, registration)
