import sys
import numpy as np
from scipy import linalg as la

##
# Main: solveSCLS_expGrad()
#
# Inputs:
#   - G: KxK precomputed matrix, which is U'U (U: NxK matrix)
#   - f: Kx1 precomputed vector, which is U'v (v: Nx1 vector)
#   - y0: Kx1 initial solution vector (default = (1/K, ..., 1/K))
#   - T: the maximum iterations (default = 500)
#   - tolerance: the stopping criteria which is a duality gap between (default = 0.00001)
#
# Outputs:
#   - y: Kx1 column non-negtive vector whose 1-norm is equal to 1
#   - isConverged: 1 if the algorithm converges, 0 otherwise
#
# Remarks:
#   - This function finds a least-square solution y that minimizes ||Uy - v||^2 
#     with the simplex constraint.
#   - Users should feed the precomputed invarainats rather than inputting U and v.
#   - One can easily extend the objective function to ||Uy - v||^2 + <Lambda, yy^T - V>_F
#     just by feeding an augmented F.
#
def solveSCLS_expGrad(G, f, y0=None, T=500, tolerance=0.00001, eta=50.0):

	if y0 is None:
		K = G.shape[0]
		y = (1.0/K)*np.ones((K,1))
	else:
		y = y0


	isConverged = 0

	# compute the gradient vector using the invariant parts
	# f(y) = ||Hy - h||^2 = y'H'Hy - 2(h'H)y + ||b||^2 (positive scalar)
	# grad(f) = 2H'Hy - 2(h'H)' = 2(H'Hy - H'h) (Kx1 column vector)
	gradients = 2.0*(G.dot(y) - f)

	# perform update steps until convergence
	for t in range(T):

		# step 1: perform component-wise multiplicative update in the original space
		y = np.multiply(y, np.exp(-eta*gradients))

		# step 2: project onto the K-dimensional simplex
		y = y / la.norm(y, ord=1)

		# step 3: evaluate the gradient
		gradients = 2.0*(G.dot(y) - f)

		# step 4: compute the duality gap and check the convergence
		dualityGap = (gradients - np.min(gradients)).T.dot(y)
		if (dualityGap < tolerance).all():
			# If every component of duality gap is less than the tolerance,
			isConverged = 1
			break

	return y, isConverged

##
# Main: solveSCLS_admmDR()
#
# Inputs: 
#   - G: KxK precomputed matrix, which is inv(gamma*U'U + I_k) (U: NxK matrix)
#   - f: Kx1 precomputed vector, which is gamma*U'U (v: Nx1 vector)
#   - y0: Kx1 initial solution vector (default = (1/K, ..., 1/K))
#   - T: the maximum iterations (default = 500)
#   - tolerance: the stopping criteria which is 2-norm change in consecutive solutions (default = 0.00001)
#
# Outputs:
#   - y: Kx1 column non-negtive vector whose 1-norm is equal to 1
#   - isConverged: 1 if the algorithm converges, 0 otherwise
#
# Remarks:
#   - This function finds a least-square solution y that minimizes ||Uy - v||^2 
#     with the simplex constraint.
#   - Users should feed the precomputed invarainats rather than inputting U and v.
#   - One can easily extend the objective function to ||Uy - v||^2 + <Lambda, yy^T - V>_F
#     just by feeding an augmented F.
#
def solveSCLS_admmDR(G, f, y0=None, T=500, tolerance=0.00001, L=1.9):

	if y0 is None:
		K = G.shape[0]
		y = (1.0/K)*np.ones((K,1))
	else:
		y = y0

	
	isConverged = 0
	b = y

	prox = lambda x: G.dot(x + f)
	for _ in range(T):
		prev_y = y
		a = prox(2*y - b)
		b = b + L*(a - y)
		y = projectToSimplex(b)
		if la.norm(y - prev_y, 2) < tolerance:
			isConverged = 1
			break


	return y, isConverged

def projectToSimplex(y):

	u = np.sort(y)[::-1]
	one_minus_cumsum_u = 1 - np.cumsum(u)
	J = np.arange(1,len(y)+1).reshape(y.shape)
	candidates = u + np.multiply(1.0/J, one_minus_cumsum_u)
	rho = np.where(candidates > 0)[0][-1]

	one_minus_cumsum_u = np.squeeze(np.asarray(one_minus_cumsum_u))
	L = (1.0/(rho+1))*((one_minus_cumsum_u.flatten())[rho])
	x = np.maximum(y+L, 0)

	return x


##
# Inner: solveSCLS_activeSet(A, b)
#
# Remark:
#   - Active set solver for the simplex-constrained least squares problem
#
#         minimize norm(A*x-b)^2/2 s.t. x >= 0 and sum(x) = 1
#
#   - At exit, should satisfy A'*(b-A*x) + w - l = 0 to roughly machine precision.
#
def solveSCLS_activeSet(A, b):

	# Reduce to a square simplex problem
	n = A.shape[1]
	Q, R = la.qr(A, mode='economic')
	b = Q.T.dot(b)
	if len(b.shape) < 2:
		b = b[:, np.newaxis]

	# First k indices in permutation p are in the passive set; start empty
	p = np.array(range(n))
	k = 0

	# Initial guess and dual
	x = np.zeros((n,1))
	r = R.T.dot(b - R.dot(x))
	w = r - x.T.dot(r)
	

	# Max inner iterations allowed
	it = 0
	itmax = 30*n

	# Tolerance for step zero convergence
	normR1 = la.norm(R, 1)
	normRinf = la.norm(R, np.inf)
	eps = sys.float_info.epsilon
	tol = 2*n*eps*normRinf*la.norm(b, np.inf)

	# Outer loop: add free variables
	while k < n and (w[k:] > 0).any():

		# Move index with largest dual into the passive set
		t = np.argmax(w[k:n])
		p, x, w, R, b = givUpdate(k+t, k, p, x, w, R, b)

		k += 1

		# Figure out where we would like to go next
		c, _, _, _ = la.lstsq(R[:k,:k].T, np.ones((k,1)))
		l = (1 - c.T.dot(b[:k])) / (c.T.dot(c))
		s, _, _, _ = la.lstsq(R[:k,:k], b[:k] + (l.item()*c))


		# Inner loop to add constraints
		while (s <= 0).any() and it < itmax:
			it = it + 1

			# Find step size and the constraint to activate
			QQ = np.where(s <= 0)
			if any(x[QQ] <= 0):
				alpha = 0
				t = np.argmin(x[QQ])
			else:
				tmp = np.divide(x[QQ], (x[QQ]-s[QQ]))
				alpha = np.min(tmp)
				t = np.argmin(tmp)
				t  = QQ[0][t]

			# Move to the first binding constraint (x(t) = 0)
			x[:k] = x[:k] + alpha*(s - x[:k])
			x[t] = 0

			# Move index t into the active set

			p, x, w, R, b = givDowndate(t, k, p, x, w, R, b)
			k = k - 1

			# Recompute s with new constraint set
			c, _, _, _ = la.lstsq(R[:k,:k].T, np.ones((k,1)))
			l = (1 - c.T.dot(b[:k])) / (c.T.dot(c))
			s, _, _, _ = la.lstsq(R[:k,:k], b[:k] + (l.item()*c))

		x[:] = 0
		x[:k] = s
		r = R.T.dot(b - R.dot(x))
		w = r - x.T.dot(r)
		w[:k] = 0

		normR1 = la.norm(R, 1)
		normRinf = la.norm(R, np.inf)
		tol = 2*n*eps*normR1*(2*normRinf*la.norm(x, np.inf) + la.norm(b, np.inf))

	x_tmp = np.zeros(x.shape)
	w_tmp = np.zeros(w.shape)
	x_tmp[p] = x
	w_tmp[p] = w
	x = x_tmp
	w = w_tmp

	return x, w, l


def givUpdate(t, k, p, x, w, R, b):

	for j in range(t-1, k-1, -1):
		p, x, w, R, b = givSwap(j, p, x, w, R, b)

	return p, x, w, R, b

def givDowndate(t, k, p, x, w, R, b):

	for j in range(t, k-1):
		p, x, w, R, b = givSwap(j, p, x, w, R, b)

	return p, x, w, R, b

def givSwap(k, p, x, w, R, b):

	p[k], p[k+1] = p[k+1], p[k]
	x[k], x[k+1] = x[k+1], x[k]
	w[k], w[k+1] = w[k+1], w[k]
	R[:,[k, k+1]] = R[:,[k+1, k]]
	G = givens(R[k,k], R[k+1,k])
	R[[k,k+1],k:] = G.dot(R[[k,k+1],k:])
	b[[k,k+1]] = G.dot(b[[k,k+1]])
	R[k+1,k] = 0

	return p, x, w, R, b

def givens(x, y):

	r = np.sqrt(x**2 + y**2)
	c = -x / r
	s = -y / r

	G = np.array([[c, s],[-np.conj(s), c]])

	return G