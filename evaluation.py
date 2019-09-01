import numpy as np
from scipy import linalg as la
import math
import sys
import time

def generateTopWords(S, B, L, dict_filename, output_filename="topWords"):


		print('[evaluation.generateTopWords] Start generating top contributing objects...')

		# Read the mapping dictionary
		with open(dict_filename, 'r') as dictFile:
			dictionary = dictFile.readlines()
			dictionary = [word.strip('\n') for word in dictionary]

		print('- Index-object mapping dictionary is properly loaded.')

		# Sort each group by the decreasing order of contributions and initialize.
		t = time.time()

		N, K = B.shape
		L = min(N, L)

		# Write the top M contributing members for each group.
		horOutputFile = open(output_filename + '.hor', 'w')
		verOutputFile = open(output_filename + '.ver', 'w')
		for k in range(K):

			B_sorted = np.sort(B[:,k])[::-1]
			I = np.argsort(B[:,k])[::-1]

			horOutputFile.write('%20s\t' % ('[%s]' % dictionary[S[k]]))
			verOutputFile.write('[%s]\n' % dictionary[S[k]])

			for l in range(L):
				horOutputFile.write(' %s' % dictionary[I[l]])
				verOutputFile.write('\t%5d: %s (%.6f)\n' % (I[l], dictionary[I[l]], B_sorted[l]))

			horOutputFile.write('\n')
			verOutputFile.write('\n')

		horOutputFile.close()
		verOutputFile.close()
		elapsedTime = time.time() - t

		# Print out the final status
		print('+ Finish generating top contributing objects!')
		print('  - Both horizontal/vertical files are generated!')
		print('  - Elapsed seconds = %.4f\n' % elapsedTime)

def evaluateClusters(B, A, S, Btilde, Cbar, C_rowSums, C, D1, D2):

	# Setup the dissimilarity measure to use.
	clusterDissimilarity_soft = clusterDissimilarity_symKL
	#clusterDissimilarity_soft = clusterDissimilarity_cos
	#clusterDissimilarity_soft = clusterDissimilarity_Fisher

	RE, RE_std   = recoveryError(Cbar, S, Btilde)
	DL           = distributionLegality(A)
	MV           = marginalValidity(B, A, Btilde, C_rowSums)
	AE1, AE2     = approximationError(C, B, A)
	DD, DD_std   = diagonalDominancy(A)

	NE, NE_std   = normalizedEntropy(Btilde)
	CP, CP_std   = clusterSparsity(B)
	CS, CS_std   = clusterSpecificity(B, C_rowSums)
	CDh, CDh_std = clusterDissimilarity_hard(B, 20)
	CDs, CDs_std = clusterDissimilarity_soft(B)
	CC, CC_std   = clusterCoherence(B, D1, D2, 20)

	BRh, BRh_std = basisRank_hard(B, S)
	BRs, BRs_std = basisRank_soft(B, S)
	BQh, BQh_std = basisQuality_hard(B, S)
	BQs, BQs_std = basisQuality_soft(B, S)

	title = '%14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s %14s' \
		% ('Recovery', 'Legality', 'Validity', 'Approximation', 'OffDiagApprox', 'Dominancy', 'Entropy', 'Specificity', 'Dissimilarity', 'SoftDissimilar', 'Coherence', 'BasisRank', 'SoftBasisRank', 'BasisQuality', 'SoftBasisQual', 'Sparsity')
	value = '%14.6f %14.4f %14.4f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14.4f %14.4f %14.2f %14.6f %14.6f' \
		% (RE, DL, MV, AE1, AE2, DD, NE, CS, CDh, CDs, CC, BRh, BRs, BQh, BQs, CP)
	stdev = '%14.6f %14s %14s %14s %14s %14.6f %14.6f %14.6f %14.6f %14.6f %14.3f %14.4f %14.4f %14.2f %14.6f %14.6f' \
		% (RE_std, 'NA', 'NA', 'NA', 'NA', DD_std, NE_std, CS_std, CDh_std, CDs_std, CC_std, BRh_std, BRs_std, BQh_std, BQs_std, CP_std);
	
	return title, value, stdev


##
# Inner: recoveryError()
#
# Remarks:
#  - This function evaluates the intrinsic quality of the matrix B
#  - This function computes the mean recovery error by averaging the
#    residual from NNLS recovery for each non-basis object in terms of 
#    basis vectors.
#  - The small is better as closer to 0
#
def recoveryError(Cbar, S, Btilde):

	# compute the recovery errors in a batch
	Cbar_S = Cbar[S.astype(int),:]
	residuals = Btilde.T.dot(Cbar_S) - Cbar


	# measure the sum of row-wise norm of residuals
	# (averaged across N objects)
	errors = np.sqrt((abs(residuals)**2).sum(axis=1))
	value = np.mean(errors)
	stdev = np.std(errors)

	return value, stdev

##
# Inner: distributionLegality()
# Remark: 
#  - This function evaluates the intrinsic quality of the matrix A
#  - This function computes the sum of all entries of A
#  - The small is better as closer to 1
#
def distributionLegality(A):
	# measure the sum of all entries of A
	value = A.sum();

	return value

##
# Inner: marginalValidity()
# Remark: 
#  - This function evaluates the intrinsic quality between B and A
#  - This function computes the difference between marginals p(z) computed
#    by summing A row-wisely and marginals computed indirectly via Bayes rule
#  - The small is better as closer to 0
#
def marginalValidity(B, A, Btilde, C_rowSums):
	# compute the row-wise marginal distribution of A
	A_rowSums = A.sum(axis=1)

	# compute the marginal distribution of A via Bayes rule
	K = B.shape[1]
	bayesResults = np.divide(np.multiply(Btilde, np.outer(np.ones((K, 1)), C_rowSums)), B.T)
	bayesResults[bayesResults == 0] = np.nan
	p_z = np.nanmean(bayesResults, axis=1)
	value = symmetricKL(A_rowSums, p_z)

	return value

##
# Inner: approximationError()
# Remark: 
#  - This function evaluates the overall intrinsic quality of B and A
#  - This function computes how close the reconstrution is to the original
#    co-occurrence matrix C entre-wisely in terms of Frobenius norm
#  - The second output measures reconstruction error without diagonal entries
#  - The small is better as closer to 0
#
def approximationError(C, B, A):
	# measure how much the decomposition is close to the original
	C_prime = B.dot(A).dot(B.T)
	value1 = la.norm(C - C_prime, 'fro')
	# Python does not support Frobenius norm for vectors, thus replaced with order 2
	diagError = la.norm(np.diag(C) - np.diag(C_prime), 2)
	value2 = np.sqrt(value1**2 - diagError**2)

	return value1, value2

##
# Inner: diagonalDominancy()
# Remark: 
#  - This function evaluates the overall comparative quality of A
#  - This function computes how large the diagonal elements of A are with
#    respect to non-diagonal entries
#  - The small is better (not necessarily close to 0)
#
def diagonalDominancy(A):
	# measure how large diagonal entries are relative to other entries
	A_rowSums = A.sum(axis=1)
	dominancies = np.diag(A) / A_rowSums
	value = np.mean(dominancies)
	stdev = np.std(dominancies)

	return value, stdev

##
# Inner: normalizedEntropy()
# Remark: 
#  - This function evaluates the extrinstic quality of Btilde
#  - This function computes how much each word is concentrated on several
#    topics. If the clusters are not mature enough, cluster preference for
#    each word is close to the uniform distribution.
#  - The large is better (averaged across N objects)
def normalizedEntropy(Btilde):
	# measure how far p(z|x) is concentrated on several topics relative to the uniform distribution
	K = Btilde.shape[0]
	entropies = entropy(Btilde) / (np.log(K) / np.log(2))
	value = np.mean(entropies)
	stdev = np.std(entropies)

	return value, stdev

##
# Inner: clusterSparsity()
# Remark:
#   - This function evaluates how sparse each cluster is.
#   - The higher is better (averaged across K clusters).
def clusterSparsity(B):

	N = B.shape[0]
	colNorms_l1 = np.abs(B).sum(axis=0)
	colNorms_l2 = np.sqrt((B**2).sum(axis=0))
	sparsities = (np.sqrt(N) - (np.divide(colNorms_l1, colNorms_l2))) / (np.sqrt(N)-1.0)
	value = np.mean(sparsities)
	stdev = np.std(sparsities)

	return value, stdev

##
# Inner: clusterSpecificity()
# Remark: 
#  - This function evaluates the extrinsic quality of B
#  - This function computes how far each cluster is from the corpus
#    distribution. Note that ill-conditioned clustering (e.g: not-enough 
#    clusters) often yields the clusters similar to the corpus distribution.
#  - The larger is better (averaged across K clusters)
#
def clusterSpecificity(B, C_rowSums):
	# measure how much each cluster is far from the corpus distribution
	divergences = divergenceKL(B, C_rowSums)
	value = np.mean(divergences)
	stdev = np.std(divergences)

	return value, stdev

##
# Inner: clusterDissimilarity_hard()
# Remark: 
#  - This function evaluates the extrinsic quality of B
#  - This function computes how different each cluster is from other
#    clusters via counting the number of unique objects that appear in the
#    target cluster, but do not appear in all other clusters in top words.
#  - The larger is better (averaged across K clusters)
#
def clusterDissimilarity_hard(B, M):
	# sort each group by the decreasing order of contributions
	I = np.argsort(B, axis=0)[::-1]

	# pick the indices of top M contributing objects
	N, K = B.shape
	M = min(M, N)
	I = I[0:M,:]

	# prepare variables
	colSet = set(range(K))
	dissimilarities = np.zeros(K)

	# count the number of unique objects in top M contributions
	for k in range(K):
		currentObjects = set(I[:,k])
		otherObjects = np.unique(I[:, list(colSet.difference({k}))])
		uniqueObjects = currentObjects.difference(otherObjects)
		dissimilarities[k] = len(uniqueObjects)

	# measure how many of top M objects do not appear in other cluster's top M objects
	value = np.mean(dissimilarities)
	stdev = np.std(dissimilarities)

	return value, stdev

##
# Inner: clusterDissimilarity_symKL()
# Remark: 
#  - This function evaluates the extrinsic quality of B
#  - This function computes how different each cluster is from other
#    clusters via measuring symmetric KL-divergences between the target
#    cluster and all other clusters.
#  - The larger is better (averaged across K clusters)
#
def clusterDissimilarity_symKL(B):
	# prepare variables
	K = B.shape[1]
	colSet = set(range(K))
	dissimilarities = np.zeros(K)

	# evaluate the symmetric KL-divergence between one and the others
	for k in range(K):
		currentClusters = B[:, k]
		otherClusters = B[:, list(colSet.difference({k}))]
		dissimilarities[k] = np.mean(symmetricKL(currentClusters, otherClusters))

	# measure how much each cluster is different from all other clusters
	value = np.mean(dissimilarities)
	stdev = np.std(dissimilarities)

	return value, stdev

##
# Inner: clusterDissimilarity_cos()
# Remark: 
#  - This function evaluates the extrinsic quality of B
#  - This function computes how different each cluster is from other
#    clusters via measuring the cosine similarities between the target
#    cluster and all other clusters.
#  - The larger is better (averaged across all pairwise comparison)
#
def clusterDissimilarity_cos(B):

	K = B.shape[1]
	BtB = B.T.dot(B)
	offDiagonals = set(range(K**2)).difference(set(range(K,K**2, K)))
	dissimilarities = BtB.reshape(BtB.size)[list(offDiagonals)]
	value = np.mean(dissimilarities)
	stdev = np.std(dissimilarities)

	return value, stdev

##
# Inner: clusterDissimilarity_Fisher()
# Remark: 
#  - This function evaluates the extrinsic quality of B
#  - This function computes how different each cluster is from other
#    clusters via measuring the Fisher distance between the target
#    cluster and all other clusters.
#  - The larger is better (averaged across all pairwise comparison)
#
def clusterDissimilarity_Fisher(B):

	K = B.shape[1]
	sqrtB = np.sqrt(B)
	sqrtBtB = sqrtB.T.dot(sqrtB)
	offDiagonals = set(range(K**2)).difference(set(range(K,K**2, K)))
	dissimilarities = math.acos(sqrtBtB.reshape(sqrtBtB.size)[list(offDiagonals)])
	value = np.mean(dissimilarities)
	stdev = np.std(dissimilarities)

	return value, stdev

##
# Inner: clusterCoherence()
# Remark: 
#  - This function evaluates the extrinsic quality of B
#  - This function counts how many strange word pairs (occuring alone in
#    many examples, but rarely co-occuring together in training examples)
#    exist in the top words of each cluster.
#  - The larger close to 0 is better (averaged across all clusters)
#
def clusterCoherence(B, D1, D2, L):

	# sort each group by the decreasing order of contributions
	I = np.argsort(B, axis=0)[::-1]

	# prepare variables
	K = B.shape[1]
	coherences = np.zeros(K)

	# find the coherence of each cluster
	epsilon = 0.01
	for k in range(K):
		for i in range(1, L):
			top_i = I[i, k]
			for j in range(i):
				# smoothe the numerator by adding epsilon to avoid taking the logarithm of zero
				top_j =I[j, k]
				coherences[k] = coherences[k] + np.log((D2[top_i,top_j] + epsilon)/D1[top_j])

	value = np.mean(coherences)
	stdev = np.std(coherences)

	return value, stdev

##
# Inner: basisRank_hard()
# Remark: 
#  - This function evaluates the extrinsic quality of S with respect to B
#  - This function computes the average rank of basis object in each cluster
#  - No specific criteria (averaged across all clusters)
#
def basisRank_hard(B, S):

	# sort each group by the decresing order of contributions
	I = np.argsort(B, axis=0)[::-1]

	# prepare variables
	K = B.shape[1]
	hardRanks = np.zeros(K)

	# find the rank of basis in each cluster
	for k in range(K):
		hardRanks[k] = np.where(I[:,k] == S[k])[0]

	# measure the mean rank
	value = np.mean(hardRanks)
	stdev = np.std(hardRanks)

	return value, stdev

##
# Inner: basisRank_soft()
# Remark: 
#  - This function evaluates the extrinsic quality of S with respect to B
#  - This function computes the average log-likelihood difference between
#    the top object and basis object in each cluster
#  - No specific criteria (averaged across all clusters)
#
def basisRank_soft(B, S):

	# sort each group by the decreasing order of contributions
	B_sorted = np.sort(B, axis=0)[::-1]

	# compute the log ratio
	softRanks = np.log(B_sorted[0,:]) - np.log(np.diag(B[S,:]))

	# measure the mean ratio
	value = np.mean(softRanks)
	stdev = np.std(softRanks)

	return value, stdev

##
# Inner: basisQuality_hard()
# Remark: 
#  - This function evaluates the extrinsic quality of S with respect to B
#  - This function computes the average rank of the basis object of target
#    cluster in other clusters.
#  - The higher is better (averaged across all clusters)
#
def basisQuality_hard(B, S):

	# sort each group by the decreasing order of contributions
	I = np.argsort(B, axis=0)[::-1]

	# prepare variables
	N, K = B.shape
	colSet = set(range(K))
	offset = np.dot(N,range(K-1))
	hardQualities = np.zeros(K)

	# find the rank of the current basis in all other clusters
	for k in range(K):
		otherObjects = I[:, list(colSet.difference({k}))]
		hardRanks = np.where(otherObjects.reshape(-1, order='F') == S[k]) - offset
		hardQualities[k] = np.mean(hardRanks)

	# measure the mean quality
	value = np.mean(hardQualities)
	stdev = np.std(hardQualities)

	return value, stdev

##
# Inner: basisRank_soft()
# Remark: 
#  - This function evaluates the extrinsic quality of S with respect to B
#  - This function computes the average likelihood of the basis object in
#    all other clusters.
#  - The smaller (must be close to 0) is better (averaged across all clusters)
#
def basisQuality_soft(B, S):

	# prepare variables
	K = B.shape[1]
	colSet = set(range(K))
	softQualities = np.zeros(K)

	# find the probability of the current basis in all other clusters
	for k in range(K):
		otherClusters = colSet.difference({k})
		softRanks = B[S[k], list(otherClusters)]
		softQualities[k] = np.mean(softRanks)

	# measure the mean quality
	value = np.mean(softQualities)
	stdev = np.std(softQualities)

	return value, stdev

##
# Helper: divergenceKL()
# In/Outs: every computation is done between column vectors
#   - p: vector / q: vector --> return positive real: D(p || q)
#   - p: vector / q: matrix --> return row vector: D(p || each column of q)
#   - p: matrix / q: vector --> return row vector: D(each column of p || q)
#   - p: matrix / q: matrix --> return row vector: D(each column of p || each column of q)
#
def divergenceKL(p, q):

	# get the sizes and perform sanity check
	if len(p.shape) == 1:
		p = p[:,None]
	if len(q.shape) == 1:
		q = q[:,None]

	p_rows, p_cols = p.shape
	q_rows, q_cols = q.shape

	if p_rows != q_rows:
		raise ValueError('* Incomparable distributions!')

	# smoothe the entries of q if they are exactly zero to avoid infinity
	q[q==0] = sys.float_info.epsilon

	# perform column-normalization
	q_colSums = q.sum(axis=0)
	q = q / q_colSums

	# horizontally duplicate column vectors if the sizes do not match
	if p_cols > q_cols:
		q = np.tile(q, (1, p_cols / q_cols))
	elif p_cols < q_cols:
		p = np.tile(p, (1, q_cols / p_cols))

	# compute the divergence in terms of bit unit with the log base 2
	# note that 0*log0 = NaN whereas its limit value is 0
	divergences = (np.multiply(p, np.log(np.divide(p, q))) / np.log(2))
	divergences[np.isnan(divergences)] = 0

	values = divergences.sum(axis=0)

	return values[0]

def divergenceKL2(P,Q):

	# get the sizes and perform the sanity check
	P_rows, P_cols = P.shape
	Q_row, Q_cols = Q.shape

	if P_rows != Q_rows or P_cols != Q_cols:
		raise ValueError('* Incomparable distributions!')

	# smoothe the entries of q if they are exactly zero to avoid infinity
	Q[Q==0] = sys.float_info.epsilon

	# perform column-normalization
	Q = Q / sum(sum(Q))

	# compute the divergence in terms of bit unit with the log base 2
	# note that 0*log0 = NaN whereas its limit value is 0
	divergences = (p * np.log(p/q)) / np.log(2)
	divergences[np.isnan(divergences)] = 0
	
	values = divergences.sum()

	return values

##
# Helper: symmetricKL()
# In/Outs: every computation is done between column vectors
#   - p: vector / q: vector --> return positive real (1 vs 1)
#   - p: vector / q: matrix --> return row vector (1 vs each column)
#   - p: matrix / q: vector --> return row vector (each column vs 1)
#   - p: matrix / q: matrix --> return row vector (each column vs each column)
#
def symmetricKL(p, q):

	values = 0.5*(divergenceKL(p, q) + divergenceKL(q, p))

	return values

def symmetricKL2(P, Q):

	values = 0.5*(divergenceKL2(P, Q) + divergenceKL2(Q, P))

	return values

##
# Helper: fisherDistance()
# In/Outs: every computation is done between column vectors
#   - p: vector / q: vector --> return positive real (1 vs 1)
#   - p: vector / q: matrix --> return row vector (1 vs each column)
#   - p: matrix / q: vector --> return row vector (each column vs 1)
#   - p: matrix / q: matrix --> return row vector (each column vs each column)
#
def fisherDistance(p, q):

	# get the sizes and perform sanity check
	p_rows, p_cols = p.shape
	q_rows, q_cols = q.shape
	if p_rows != q_rows:
		raise ValueError('* Incomparable distributions!')

	# horizontally duplicate column vectors if the sizes do not match
	# note that each entry becomes its square root first
	if p_cols > q_cols:
		q = np.sqrt(np.tile(q, (1, p_cols / q_cols)))
	elif p_cols < q_cols:
		p = np.sqrt(np.tile(p, (1, q_cols / p_cols)))

	# compute the Fisher information metric
	distances = sum(p*q, axis=0)
	values = math.acos(distances)

	return values

##
# Helper: entropy()
# In/Outs:
#   - p: vector --> real value
#   - p: matrix --> return row vector (entropy of each column vector)
#
def entropy(p):
	# remember the position where the entries of p is exactly zero
	# this is because eps*log(eps/positive) -> 0 as eps -> 0
	I = (p == 0)

	# compute the entropies in terms of bit unit with the log base 2
	# note that 0*log0 = NaN whereas its limit value is 0
	entropies = -(p * np.log(p)) / np.log(2)
	entropies[I] = 0
	values = entropies.sum(axis=0)

	return values