import time
import numpy as np
import sys
from scipy import linalg as la
from scipy.sparse import linalg as spla
from pyJSMF.optimization import solveSCLS_expGrad, solveSCLS_admmDR, solveSCLS_activeSet, projectToSimplex

##
# Main: findS()
#
# Inputs:
#   - Cbar: NxN row-normalized co-occurrence matrix (row-stochastic)
#   - K: the number of basis vectors / the low-dimension for pca/tsne
#   + option: method to find basis vectors (default = 'sparsePartial')
#
# Outputs:
#   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
#   - diagR: 1xK vector indicating the scores of each basis vector
#   - elapsedTime: total elapsed amount of seconds
#
# Remarks: 
#   - This function performs QR-factorization with the row-pivoting, 
#     finding the given number of approximate nonnegative basis vectors.
#   - The 'full' method is useful only when N is small.
#  
def findS(Cbar, K, option='sparsePartial'):

	Cbar_t = Cbar.T

	print('[inference.findS] Start finding the set of anchor bases S...')

	t = time.time()
	if option == 'densePartial':
		S, diagR = densePartialQR(Cbar_t, K)
	elif option == 'sparsePartial':
		S, diagR = sparsePartialQR(Cbar_t, K)
	elif option == 'full':
		S, diagR = fullQR(Cbar_t, K)
	else:
		raise ValueError('  * Undefined option [%s] is given!' % option)
	elapsedTime = time.time()-t

	print('+ Finish finding set S!')
	print('  - Discovered %i basis vectors by [%s] method.' % (K, option))
	print('  - Elapsed time = %.4f seconds\n' % elapsedTime)

	return S.astype(int), diagR, elapsedTime


##
# Inner: densePartialQR()
#
# Inputs:
#   - P: NxN column-normalized co-occurrence matrix (column-stochastic)
#   - K: the number of basis vectors 
#
# Outputs:
#   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
#   - diagR: 1xK vector indicating the distance to the current subspace
#
# Remarks: 
#   - This function greedily selects K column vectors of P by using
#     Gram-Schmidt process, which is equivalent to column-pivoting.
#   - It does not fully factorize P into QR, but project every non-basis
#     rows of P, loosing the sparse structure of P.
#
def densePartialQR(P, K):

	# compute squared sums for each column and prepare return variables
	colSquaredSums = np.sum(P**2, axis=0)
	S = np.zeros(K)
	diagR = np.zeros(K)
	isColBasis = np.zeros(P.shape[1], dtype=bool)

	for k in range(K):
		# find the farthest column vector from the origin
		maxSquaredSum = np.max(colSquaredSums)
		maxCol = np.argmax(colSquaredSums)
		S[k] = maxCol
		diagR[k] = np.sqrt(maxSquaredSum)
		isColBasis[maxCol] = True

		# normalize the column vector corresponding to the current basis
		P[:,maxCol] = P[:,maxCol] / diagR[k]

		# project all other columns down to the orthogonal complement of
		# the subspace spanned by the current set of basis vectors
		innerProducts = P[:,maxCol].dot(P)
		P = P - np.outer(P[:,maxCol],innerProducts)

		# recomputer the squared sums for every column (without exclusion)
		# ensure that the selected basis vectors are never chosen again
		# (theoretically not necessary, but numerically safer)
		colSquaredSums = np.sum(P**2, axis=0)
		colSquaredSums[isColBasis] = False

	return S, diagR

##
# Inner: sparsePartialQR()
#
# Inputs:
#   - P: NxN column-normalized co-occurrence matrix (column-stochastic)
#   - K: the number of basis vectors 
#
# Outputs:
#   - S: 1xK vector having the indices corresponding to K approximate nonnegative basis vectors
#   - diagR: 1xK vector indicating the distance to the current subspace
#
# Remarks: 
#   - This function greedily selects K column vectors of P by using
#     Gram-Schmidt process, which is equivalent to column-pivoting.
#   - It does not fully factorize P into QR implicitly updating Q instead of
#     changing P (without losing the sparse structure of P).
#
def sparsePartialQR(P, K):

	# computer squared sums for each column and prepare return variables
	colSquaredSums = np.sum(P**2, axis=0)
	S = np.zeros(K)
	diagR = np.zeros(K)
	isColBasis = np.zeros(P.shape[1], dtype=bool)

	Q = np.zeros((P.shape[0], K))
	for k in range(K):
		# find the farthest column vector from the origin
		maxSquaredSum = np.max(colSquaredSums)
		maxCol = np.argmax(colSquaredSums)
		S[k] = maxCol
		diagR[k] = np.sqrt(maxSquaredSum)
		isColBasis[maxCol] = True

		# compute the next basis q_n
		Q[:,k] = P[:,maxCol]
		if k > 0:
			# sumProjections = sum_{j=1}^{k-1} proj_{e_j}(p_max)
			sumProjections = Q[:,0:k].dot(Q[:,0:k].T).dot(P[:,maxCol])

			# q_k = p_max - sum of projections to each basis
			Q[:,k] = Q[:,k] - sumProjections

		# normalize the column vector corresponding to the current basis
		Q[:,k] = Q[:,k] / diagR[k]

		# update the squared sums of column vector implicitly
		# || p_j - <e, p_j>e ||^2 = <p_j, p_j> - 2<e, p_j><e, p_j> + <e, p_j><e, p_j><e, e> = <p_j, p_j> - <e, p_j>^2
		# the update factor becomes collectively (e'*P).^2
		colSquaredSums = colSquaredSums - (Q[:,k].T.dot(P))**2
		colSquaredSums[isColBasis] = False

	return S, diagR

def fullQR(P, K):
	# perform full QR-factorization with column-pivoting
	Q, R, S = la.qr(P, pivoting=True)
	diagR = abs(np.diag(R))

	# extract values corresponding to only the given number of basis vectors
	S = S[range(K)]
	diagR = diagR[range(K)]


#def compressPCA(Cbar, dimension):
	# TODO


#def compressTSNE(Cbar, dimension):
	# TODO


#def influenceScore(P, K):
	# TODO

##
# Main: recoverB()
#
# Inputs:
#   - Cbar: NxN row-normalized co-occurrence matrix
#   - C_rowSums: Nx1 vector having sums of each row in original C matrix
#   - S: 1xK vector having the row indices of approximate basis vectors
#
# Intermediates:
#   - H: NxK matrix having where each column is a basis vector (H = Cbar_S')
#   - h: Nx1 column vector indicating a non-basis vector 
#   - y: Kx1 column vector, non-negative least square solution in the simplex
#
# Outputs:
#   - B: NxK tall matrix where B_{nk} = p(x=n | z=k)
#   - Btilde: KxN fat matrix where Btilde_{kn} = p(z=k | x=n)
#
def recoverB(Cbar, C_rowSums, S, option='admmDR'):

	# initialize and prepare return variables
	N = Cbar.shape[0]
	K = S.size
	B = np.zeros((N, K))
	Btilde = np.zeros((K, N))
	convergences = np.zeros(N)

	# precompute the invariant parts
	U = Cbar[S.astype(int), :].T
	Ut = U.T
	UtU = Ut.dot(U)

	# print out the initial status
	print('[inference.recoverB] Start recovering the object-cluster B...')

	# compute the Btilde (for each member object)
	t = time.time()

	# Performs the trivial inference for the basis vectors
	Btilde[:,S] = np.identity(K)
	convergences[S] = 1

	if option == 'expGrad':
		for n in range(N):
			# Skips the basis vectors.
			if n in S:
				continue

			# if the given member is not a basis vector
			v = Cbar[n,:].T
			Utv = Ut.dot(v)
			Utv = Utv.reshape((len(Utv),1))
			y, isConverged = solveSCLS_expGrad(UtU, Utv)

			# save the recovered distribution p(z | x=n) and convergence
			Btilde[:,n] = y.T
			convergences[n] = isConverged

			# print out the progress
			if n % 500 == 0:
				print('  - %i-th member...' % n)

	elif option == 'admmDR':

		gamma = 3.0

		# Precompute the invariant parts.
		G = la.inv(gamma*UtU + np.identity(K))

		for n in range(N):
			# Skips the basis vectors.
			if n in S:
				continue

			# if the given member is not a basis vector
			v = Cbar[n,:].T
			Utv = Ut.dot(v)
			y, _, _, _ = la.lstsq(UtU, Utv)
			y, isConverged = solveSCLS_admmDR(G, gamma*Utv, projectToSimplex(y))

			# save the recovered distribution p(z | x=n) and convergence
			Btilde[:,n] = y.T
			convergences[n] = isConverged

			# print out the progress
			if n % 500 == 0:
				print('  - %d-th member...' % n)


	elif option == 'activeSet':
		for n in range(N):
			# Skip the basis vectors
			if n in S:
				continue

			# If the given member is not a basis vector
			v = Cbar[n,:].T
			y, _, _ = solveSCLS_activeSet(U, v)

			# Save the recovered distribution p(z|x=n) and convergence.
			Btilde[:,n] = y.T

			# Print out tBhe progress for each set of objects.
			if n % 500 == 0:
				print('  - %d-th member...' % n)


	# recover the B (after finishing simplex NNLS)
	denominators = 1.0/(Btilde.dot(C_rowSums))
	B = (Btilde.T)*(np.outer(C_rowSums,denominators))
	elapsedTime = time.time()-t

	# print out the final status
	loss = la.norm(U.dot(Btilde) - Cbar, 'fro')
	print('+ Finish recovering B matrix using [%s]' % option)
	print('  - %i/%i objects are converged.' % (sum(convergences), N))
	print('  - loss = %.4f (By Frobenius norm).' % loss)
	print('  - Elapsed time = %.4f seconds.\n' % elapsedTime)


	return B, Btilde, elapsedTime



##
# Main: recoverA()
#
# Inputs:
#   - C: NxN original co-occurrence matrix
#   - B: NxK object-cluster matrix (column-stochastic)
#   - S: 1xK vector having the row indices of approximate basis vectors
#   - option: the method of recovery (default = 'diagonal')
#
# Outputs:
#   - A: KxK cluster-cluser matrix where A_{kl} = p(z1=k | z2=l)
#
# Remarks:
#   - This function recovers the matrix A by two different methods.
#
def recoverA(C, B, S, option='diagonal'):

	print('[inference.recoverA] Start recovering the cluster-cluster A...')

	t = time.time()
	if option == 'diagonal':
		A = diagonalRecovery(C, B, S)
	elif option == 'pseudoInverse':
		A = pseudoInverseRecovery(C, B)
	#elif option == 'optimize':
	#	A = optimizeRecovery(C, B, S)
	else:
		raise ValueError('  * Undefined option [%s] is given!' % option)

	elapsedTime = time.time() - t

	print('+ Finish recovering A!')
	print('  - [%s] recovery is used.' % option)
	print('  - Elapsed time = %.4f seconds.\n' % elapsedTime)

	return A, elapsedTime


def diagonalRecovery(C, B, S):

	C_SS = C[np.ix_(S,S)]
	B_S = B[S.astype(int),:]
	invB_S = np.diag(1.0 / np.diag(B_S))
	A = invB_S.dot(C_SS).dot(invB_S)

	return A

def psuedoInverseRecovery(C, B):

	pinvB = la.pinv(B)
	A = pinvB.dot(C).dot(pinvB.T)

	return A

def optimizeRecovery(C, B, S):

	# TODO: Implement multiplicative update for approx error minimization

	return None