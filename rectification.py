import time
import numpy as np
from scipy import linalg as la
from scipy.sparse import linalg as spla

##
# Wrapper: rectifyC()
#
# Inputs:
#   - C: NxN co-occurrence matrix (joint-stochastic)
#   - K: the number of basis vectors (the number of topics)
#   - rectifier: choice of rectification method (default = "AP")
#
# Outputs:
#   - C: NxN co-occurrence matrix (joint-stochastic & doubly-nonnegative)
#   + values: 2xT statistics
#     - 1st row: changes between before and after iteration in terms of Frobenius norm
#     - 2nd row: average square difference betweeo before and after projections in terms of Frobenius norm 
#   - elapsedTime: total elapsed amount of seconds
#
# Remarks: 
#   - This function wraps multiple different algorithms for rectification.
#
def rectifyC(C, K, rectifier='AP'):

	if rectifier == 'AP':
		C, values, elapsedTime = rectifyC_AP(C, K)
	elif rectifier == 'DR':
		C, values, elapsedTime = rectifyC_DR(C, K)
	else:
		values = []
		elapsedTime = 0

	return C, values, elapsedTime

##
# Main: rectifyC_AP()
#
# Inputs:
#   - C: NxN co-occurrence matrix (joint-stochastic)
#   - K: the number of topics
#   - T: the number of maximum iterations (default = 150)
#
# Outputs:
#   - C: NxN co-occurrence matrix (joint-stochastic & doubly-nonnegative)
#
def rectifyC_AP(C, K, T=15):

	values = np.zeros((2, T))
	C_NN = C

	print('+ Start alternating projection')

	startTime = time.time()

	for t in range(T):
		# backup the previous C
		C_prev = C_NN

		# do one iteration of alternating projection
		C_PSD = nearestPSD(C_NN, K)
		d_PSD = la.norm(C-C_PSD, 'fro')
		C_JS  = nearestJS(C_PSD)
		d_JS  = la.norm(C-C_JS, 'fro')
		C_NN  = nearestNN(C_JS)
		d_NN  = la.norm(C-C_NN, 'fro')

		values[0, t] = la.norm(C_NN - C_prev, 'fro')
		values[1, t] = (d_PSD**2+d_JS**2+d_NN**2)/6

		if t%1 == 0:
			print('  - %d-th iteration... (%e / %e)' % (t+1,values[0,t],values[1,t]))


	C = C_NN/sum(sum(C_NN))
	elapsedTime = time.time()-startTime
	print('+ Finish alternating projection')
	print('  - Elapsed seconds = %.4f\n' % elapsedTime)


	return C, values, elapsedTime

##
# Main: rectifyC_DR()
#
# Inputs:
#   - C: NxN co-occurrence matrix (joint-stochastic)
#   - K: the number of topics
#   - T: the number of maximum iterations (default = 150)
#
# Outputs:
#   - C: NxN co-occurrence matrix (joint-stochastic & doubly-nonnegative)
#
def rectifyC_DR(C, K, T=15):

	values = np.zeros((2, T))
	C_3 = C

	print('+ Start cyclic Douglas-Rachford projection')

	startTime = time.time()

	for t in range(T):
		# backup the previous C
		C_prev = C_3

		# do one iteration of alternating projection
		C_1 = projectPSD_JS(C_3, K)
		d_1 = la.norm(C-C_1, 'fro')
		C_2 = projectJS_NN(C_1)
		d_2 = la.norm(C-C_2, 'fro')
		C_3 = projectNN_PSD(C_2, K)
		d_3 = la.norm(C-C_3, 'fro')

		values[0, t] = la.norm(C_3 - C_prev, 'fro')
		values[1, t] = (d_1**2+d_2**2+d_3**2)/6

		if t%1 == 0:
			print('  - %d-th iteration... (%e / %e)' % (t+1,values[0,t],values[1,t]))


	C = C_3/sum(sum(C_3))
	elapsedTime = time.time()-startTime
	print('+ Finish cyclic Douglas-Rachford projection')
	print('  - Elapsed seconds = %.4f\n' % elapsedTime)


	return C, values, elapsedTime

##
# Inner: nearestNN()
# 
# Inputs:
#   - C: NxN co-occurrence matrix
#
# Outputs:
#   - C: NxN non-negative matrix
#
def nearestNN(C):
	C = C.clip(min=0)
	return C

##
# Inner: nearestJS()
#
# Inputs:
#   - C: NxN co-occurrence matrix
#
# Outputs:
#   - C: NxN joint-stochastic matrix
#
def nearestJS(C):
	N = C.shape[0]
	C = C + (1 - C.sum())/(N**2)
	return C

##
# Inner: nearestPSD()
#
# Inputs:
#   - C: NxN co-occurrence matrix
#   - K: the number of non-negative eigenvalues to use
#
# Outputs:
#   - C: NxN positive semi-definite matrix
#
# Remarks:
#   - This function projects the given matrix into the convex set of
#     positive semidefinite matrices with the rank K
#   - Due to epsilon numerical error, it symmetrize the matrices
#
def nearestPSD(C, K):
	# find nearest positive semidefinite matrix with the rank K
	D, V = spla.eigsh(C, k=K, which='LA')
	C = V.dot(np.diag(D.clip(min=0))).dot(V.T)
	C = 0.5*(C + C.T)
	return C

##
# Remarks:
#  - Each function implemented below represents a 2-set Douglas-Rachford 
#    operator that is repeatedly used within the DR iteration scheme.
#  - Given two sets A and B, the operator is defined as (I + R_B R_A)/2
#    where R_A denotes the reflection with respect to A.

##
# Inner: projectPSD_JS
#
# Inputs: 
#   - C: NxN co-occurrence matrix
#   - K: the number of non-negative eigenvalues to use
#
# Outputs:
#   - C: NxN co-occurrence matrix
#
def projectPSD_JS(C, K):

    R_PSD = 2*nearestPSD(C, K) - C;
    R_JS = 2*nearestJS(R_PSD) - R_PSD;
    C = (C + R_JS)/2; 

    return C

##
# Inner: projectJS_NN
#
# Inputs: 
#   - C: NxN co-occurrence matrix
#
# Outputs:
#   - C: NxN co-occurrence matrix
#
def projectJS_NN(C):

    R_JS = 2*nearestJS(C) - C;
    R_NN = 2*nearestNN(R_JS) - R_JS;
    C = (C + R_NN)/2;  

    return C

##
# Inner: projectNN_PSD
#
# Inputs: 
#   - C: NxN co-occurrence matrix
#   - K: the number of non-negative eigenvalues to use
#
# Outputs:
#   - C: NxN co-occurrence matrix
#
def projectNN_PSD(C, K):
	
    R_NN = 2*nearestNN(C) - C;
    R_PSD = 2*nearestPSD(R_NN, K) - R_NN;
    C = (C + R_PSD)/2;

    return C