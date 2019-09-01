import time
import numpy as np
from pyJSMF.inference import findS
from pyJSMF.inference import recoverB
from pyJSMF.inference import recoverA
from pyJSMF.rectification import rectifyC

##
# Main: factorizeC()
#
# Inputs:
#   - C: NxN original co-occurrence matrix (must be joint-stochastic)
#   - K: the number of basis vectors (i.e., the number of topics)
#   + rectifier:
#     - 'DC': Diagonal Completion (to be added)
#     - 'AP': Alternating Projection (main method)
#     - 'DP': Dykstra Projection (to be added)
#     - otherwise: no rectification (equivalent to the vanilla Anchor Word Algorithm)
#
# Outputs:
#   - S:           1xK column vector having the basis indices
#   - B:           NxK object-cluster matrix where B_{nk} = p(X=n | Z=k) 
#   - A:           KxK cluster-cluster matrix where A_{kl} = p(Z1=k, Z2=l)
#   - Btilde:      KxN cluster-object matrix where Btilde_{kn} = p(Z=k | X=n) 
#   - Cbar:        NxN row-normalized co-occurrence matrix where Cbar_{ij} = p(X2=j | X1=i)
#   - C_rowSums:   Nx1 vector indicating the row-wise sum of C where C_rowSums_i = p(X=i)
#   - diagR:       1xK vector indicating the scores of each basis vector
#   - C:           NxN updated C matrix after the rectification step
#
# Remarks: 
#   - This function performs the overall joint-stochastic matrix factorization (a.k.a Rectified Anchor-Words Algorithm (RAWA).
#   - Run the rectification first if specified.
#   - Run the anchor-word algorithm on the rectified co-occurrence matrix.
#  
def factorizeC(C, K, rectifier='AP', optimizer='activeSet'):

	t = time.time()

	print("+ Start rectifying C...")
	C, values, elapsedTime = rectifyC(C, K, rectifier)
	print("  - Finish rectifying C! [%f]" % elapsedTime)

	# Perform row-normalization for the (rectified) co-occurrence matrix C.
	C_rowSums = C.sum(axis=1)
	Cbar = C/C_rowSums[:,None]

	# Step 1: Find the given number of bases S. 
	# (i.e., set of indices corresponding to the anchor words)
	print("+ Start finding the set of anchor bases S...")
	S, diagR, elapsedTime = findS(Cbar, K)
	print("  - Finish finding S! [%f]" % elapsedTime)

	# Step 2: Recover object-cluster matrix B. (i.e., recovers word-topic matrix)
	print("+ Start recovering the object-cluster B...")
	B, Btilde, elapsedTime = recoverB(Cbar, C_rowSums, S, optimizer)
	print("  - Finish recovering B! [%f]" % elapsedTime)

	# Step 3: Recover cluster-cluster matrix A. (i.e., recovers topic-topic matrix)
	print("+ Start recovering the cluster-cluster A...")
	A, elapsedTime = recoverA(C, B, S)
	print("  - Finish recovering A! [%f]" % elapsedTime)

	# Finish the algorithm
	elapsedTime = time.time()-t
	print("- Finish factorizing C! [%f]" % elapsedTime)

	return S, B, A, Btilde, Cbar, C_rowSums, diagR, C