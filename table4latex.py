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
	print("\\end{table}")


if __name__ == '__main__':

	bundle_list = ['ifofL', 'thprefL', 'cstL']
	registration = ['slr', 'ant4t1w', 'ant4fa']

	DSC_nn_L = np.load('DSC_nn_L.npy')
	DSC_nn_masked = np.ma.masked_equal(DSC_nn_L, 0) 
	DSC_mean_nn = np.mean(DSC_nn_masked, axis=(0,2))

	DSC_matrix2D = np.transpose(DSC_mean_nn[:,:,0])
	print_latex_table(DSC_matrix2D, bundle_list, registration)

