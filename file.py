import time
import numpy as np
from scipy import sparse

##
# Main: bows2C()
#
# Inputs:
#   - bows: X-by-3 bag-of-words matrix
#   - min_tokens: min. number of tokens for effective training examples
#
# Outputs:
#   - C: NxN dense joint-stochastic co-occurrence matrix 
#   - D1: Nx1 example frequency where D1_i = # of examples where object i occurs
#   - D2: NxN co-example frequency where D2_{ij} = # of examples where object i and j co-occurs
#
# Remark:
#   - This function converts bag-of-words to the full/dense co-occurrence and
#     example/co-example frequencies by sequentially processing each document.
#
def bows2C(bows, min_tokens):

	# Print out the initial status
	print('[file.bows2C] Start constructing dense C...')

	# Recompute the size of vocabulary by counting the unique elements in the word numbers
	N = len(np.unique(bows[:,1]))
	M = bows[-1,0]

	# Find the row numbers where each training example ends
	endRows = np.where(bows[:-1,0] != bows[1:,0])[0]
	endRows += 1
	endRows = np.append([0], endRows)
	endRows = np.append(endRows, [len(bows)])

	# Compute co-occurrence and example/co-example frequencies for each training example.
	print('- Counting the co-occurrence for each document...')
	startTime = time.time()

	C = np.zeros((N, N))
	D1 = np.zeros(N)
	D2 = np.zeros((N,N))
	for m in range(len(endRows)-1):
		# Determine the start and end rows for this document
		startRow = endRows[m]
		endRow = endRows[m+1]
		objects = bows[startRow:endRow, 1]-1
		counts = bows[startRow:endRow, 2]

		# Skip the degenerate case when the document contains only one word with a single occurrence.
		# Note that it does not happen if min_object threshold is larger than 1 when reading bows.
		numObjects = len(objects)
		numTokens = sum(counts)
		if (numObjects == 1) and (numTokens == 1):
			continue

		# Skip the current example with less than minimum counts
		if numTokens < min_tokens:
			print('  - The document %d with only %d tokens will be ignored!' % (m, numTokens))
			continue

		# Accumulate corresponding counts to co-occurrence and example/co-example frequencies.
		# Note that co-example frequency for an object can exit only when the object occurs more than once.
		normalizer = numTokens*(numTokens-1)
		C[np.ix_(objects,objects)] += (np.outer(counts,counts) - np.diag(counts)).astype(float)/normalizer
		D1[objects] += 1
		D2[np.ix_(objects, objects)] += 1 - np.diag(counts == 1)

	# Ensure the overall sum is equal to 1.0
	entrySum = C.sum()
	if (entrySum != M):
		C /= entrySum

	elapsedTime = time.time() - startTime

	# Print out the final status
	print('+ Finish constructing C and D!')
	print('  - The sum of all entries = %.6f' % (entrySum/M))
	print('  - Elapsed Time = %.4f seconds' % elapsedTime)

	return C, D1, D2

##
# Main: bows2H()
#
# Inputs:
#   - bows: X-by-3 bag-of-words matrix
#   - min_tokens: minimum number of tokens for each training example
#
# Outputs:
#   - H: NxM word-document matrix
#   - D1: Nx1 example frequency where D1_i = # of examples where object i occurs
#   - D2: NxN co-example frequency where D2_{ij} = # of examples where object i and j co-occurs
#
# Remark:
#   - This function converts bag-of-words to the full/sparse word-document matrix (double precision) 
#     and example/co-example frequencies by matrix operation as a whole.
#
def bows2H(bows, min_tokens=5):

	# Print out the initial status.
	print("[file.bows2H] Start constructing sparse H...")
	t = time.time()

	# Count the unique word indices.
	N = len(set(bows[:,1]))

	# Construct BOW matrix where each column represents a document.
	H = sparse.csc_matrix((bows[:,2], (bows[:,1]-1,bows[:,0]-1)),shape=(N,bows[-1,0]))

	# Remove documents with less number of tokens than min_tokens
	print("+ Removing the documents based on min_tokens argument...")
	H = H[:, (H.sum(axis=0) >= min_tokens).A1]

	# Compute example and co-example frequencies if necessary.
	# Note that if vocabulary size is large, D2 can exceed the memory storage.
	D1 = (H > 0).astype(float).sum(axis=1).A1
	U = (H > 0).astype(float)
	V = (H == 1).astype(float)
	D2 = U*U.T - sparse.diags(V.multiply(V).sum(axis=1).A1, 0)

	elapsedTime = time.time()-t

	print("+ Finish coustructing sparse H and D!")
	print("  - The number of documents = %d" % H.shape[1])
	print("  - Elapsed seconds = %.4f\n" % elapsedTime)

	return H, D1, D2

##
# Main: createC()
# 
# Inputs:
#   - bows_filename: name of the input file containing bag-of-words
#   - dict_filename: name of the dictionary file
#   - stop_filename: name of the corpus file containing stop words
#   - N: number of words in vocab
#   - min_objects: minimum number of objects for effective training examples
#   - min_tokens: minimum number of tokens for effective training examples
#   - output_filename: name of the output file to write stat and dict
#
# Outputs:
#   - C:  N-by-N full co-occurrence matrix
#   - D1: N-by-1 example frequency vector
#   - D2: N-by-N co-example frequency matrix
#   - dictionary: the original/curated dictonary of vocabulary
#
def createC(bows_filename, dict_filename='', stop_filename='', N=0, min_objects=3, min_tokens=5, output_filename=''):

	bows, dictionary = readBows(bows_filename, dict_filename, stop_filename, N, min_objects, output_filename)
	C, D1, D2 = bows2C(bows, min_tokens)

	return C, D1, D2, dictionary

##
# Main: readBows()
#
# Inputs:
#   + bows_filename: the name of the input file containing bag-of-objects as a certain format
#     - 1st line: the number of training examples
#     - 2nd line: the maximum object number (some intermediate numbers might not actually exist)
#     - 3rd line: the number of rows below (not used)
#     - other lines: one space-delimited triplet (example #, object #, frequency count) per each line
#   + dict_filename: the name of the dictionary file of the corpus
#     - One object per each line
#   + stop_filename: the name of the corpus file containing stop objects
#     - One object per each line
#   - N: the size of effective objects (i.e., the size of vocabulary)
#   - min_objects: the minimum number of objects for each training example to be alive
#   - output_filename: the name of the output file reporting overall processing results
#
# Intermediates:
#   - M: the number of training examples 
#   - V: the maximum possible object number 
#   - activeObjects: the mapping from the reordered object numbers [1, N] --> originals [1, V]
#   - objectMap: the inverse map from originals [1, V] --> the reordered object numbers [1, N]
#
# Outputs:
#   - bows: matrix of filtered examples where each row contains (example #, object #, frequency)
#   - dict: the new dictionary mapping object number to actual objects
#
# Remarks: 
#   + This function reads UCI formatted Bag-of-words(objects) dataset.
#     - First, eliminate the stop objects.
#     - Second, prune out objects to the most effective N words based on tf-idfs.
#     - Third, remove the training examples that have less than minimum number of objects.
#     - Fourth, reorders the object numbers by assigning consecutive integers starting from 1.
#     - Last, it trims the dictionary according to the vocabulary size N.
#  
def readBows(bows_filename, dict_filename, stop_filename, N, min_objects, output_filename):

	# Print out initial status
	print('[file.readBows] Start reading Bag-of-words dataset...')
	startTime = time.time()

	# Open the input file and read the statistics in the header
	bowsFile = open(bows_filename, 'r')
	inputLine = bowsFile.readline()
	M = int(inputLine)
	inputLine = bowsFile.readline()
	V = int(inputLine)
	bowsFile.readline()

	# Step 0: Read the Bag-of-words(objects) content
	#bows = np.loadtxt(bowsFile, delimiter=' ',
	# dtype={'names': ('col1', 'col2', 'col3'),'formats': ('i', 'i', 'i')})
	bows = np.fromregex(bowsFile, r'([\d]+)[ \t]([\d]+)[ \t]([\d]+)', dtype='int_')
	bowsFile.close()
	print('- Dataset [%s] is parsed.' % bows_filename)

	# Python doesn't have equivalent of nargout
	#if len(kwargs) >= 2 or 'dict_filename' not in kwargs.keys():
	#	return

	# Read the vocabulary dictionary if necessary
	dictionary = readObjects(dict_filename)
	print('- Dictionary [%s] is loaded.' % dict_filename)

	# Step 1: Eliminate the stop objects if requested
	
	# Read the stop objects
	stop = readObjects(stop_filename)

	# Compute the overlap between the dictionary and the list of stop objects.
	stopObjects = np.array([idx for idx in range(len(dictionary)) if dictionary[idx] in stop])
	stopObjects += 1 # Different indexing
	stopIndices = [idx for idx in range(len(bows)) if bows[idx,1] in stopObjects]

	# Trim the bows by cropping out stop indices.
	bows = np.delete(bows, stopIndices, axis=0)
	print('- Stop objects are eliminated based on [%s].' % stop_filename)

	# Check the pruning condition
	if N <= 0:
		# Exit immediately if N does not make sense
		print('* No pruning has been done!')
		return

	# Step 2: Compute the indices where each new training example starts.
	endRows = np.where(bows[:-1,0] != bows[1:,0])[0]
	endRows += 1
	endRows = np.append([0], endRows)
	endRows = np.append(endRows, [len(bows)])

	# Compute the term-frequencies, document-frequencies, and inverse document-freqencies.
	tfs = np.zeros(V, dtype=float)
	dfs = np.zeros(V, dtype=float)
	for m in range(len(endRows)-1):

		# Extract the interval of indices corresponding to each training example.
		startRow = endRows[m]
		endRow = endRows[m+1]

		# Read the list of object numbers and corresponding counts
		objects = bows[startRow:endRow, 1]
		counts = bows[startRow:endRow, 2]

		# Accumulate the term-frequencies by their count occurrences.
		tfs[objects-1] = tfs[objects-1] + counts

		# Increase the document-frequency of each object by 1.
		dfs[objects-1] += 1


	# Note that we at least prune all the object which shows up more than 50% of the documents
	# This is our document-frequency cut-off to be included in the effective vocabulary
	# Option 1: Integer flooring on df scores.
	idfs = np.log(np.floor(M/dfs))
	# Option 2: Just remove objects that occur in more than half of the documents.
	#idfs = np.log(M/dfs)
	#idfs[dfs > M/2] = 0

	# Evaluate the tf-idf scores.
	tfIdfs = tfs * idfs

	# Sort the tf-idf scores by decreasing order and filter out NaN
	# Note that V is not the vocabulary size, but the maximum possible
	# objet number. Some numbers could be unused, making tf-idf as 0/0.
	nanIndices = np.isnan(tfIdfs)
	tfIdfs[nanIndices] = 0
	indices = tfIdfs.argsort()[::-1]
	N = min(N, V - sum(nanIndices))

	# Discard every bag-of-word entries with an irrelevant object.
	# Note that actual number of objects = V - non-existing objects
	irrelevantObjects = indices[N:]
	irrelevantIndices = [idx for idx in range(len(bows)) if bows[idx,1]-1 in irrelevantObjects]
	bows = np.delete(bows, irrelevantIndices, axis=0)

	# Step 3: Remove training examples having less than the minimum number of objects.
	# Recalculate the indices where each new training example starts.
	print('+ Removing the documents based on min_objects argument...')
	endRows = np.where(bows[:-1,0] != bows[1:,0])[0]
	endRows += 1
	endRows = np.append([0], endRows)
	endRows = np.append(endRows, [len(bows)])

	# Compute the number of different objects in each examples.
	# Note that we consider the number of different types of objects, not the number of object occurrences.
	numObjects = np.diff(endRows)

	# Filter out the documents with less than min_objects threshold.
	removeExamples = [idx for idx in range(len(numObjects)) if numObjects[idx] < min_objects]
	activeIndices = [True]*len(bows)
	for idx in range(len(removeExamples)):
		m = removeExamples[idx]
		activeIndices[endRows[m]:endRows[m+1]] = [False]*(endRows[m+1]-endRows[m])
		print('  - The document %d with only %d objects will be ignored' % (bows[endRows[m],0], numObjects[m]))
	bows = bows[activeIndices,:]

	# Step 4: Reorder the remaining objects by assigning new consecutive object numbers
	activeObjects = np.unique(bows[:,1])
	N = len(activeObjects)
	objectMap = np.zeros(V)
	objectMap[indices[0:N]] = range(1,N+1)
	bows[:,1] = objectMap[bows[:,1]-1]

	print('- Reordering object number is done.')

	elapsedTime = time.time() - startTime

	# Save the statistics if output is specified
	outputFile = open(output_filename + '.stat', 'w')
	for v in range(N):
		obj = indices[v]
		outputFile.write('%6d\t%8d\t%-20s\t%.6f\n' % (v+1, obj+1, dictionary[obj], tfIdfs[obj]))
	for v in range(N+1,V):
		obj = indices[v]
		outputFile.write('%6d\t%8d\t%-20s\t%.6f\n' % (0, obj+1, dictionary[obj], tfIdfs[obj]))
	outputFile.close()
	print('- Pruning statistics [%s] is generated.' % (output_filename+'.stat'))

	# Save the curated N objects as a new vocabulary dictionary.
	dictionary = dictionary[indices[0:N]]
	outputFile = open(output_filename + '.dict', 'w')
	for n in range(N):
		outputFile.write('%s\n' % dictionary[n])
	outputFile.close()
	print('- Curated dictionary file [%s] is generated.' % (output_filename+'.dict'))


	print('+ Finish reading Bag-of-words dataset!')
	print('  - Elapsed Time = %.4f seconds' % elapsedTime)

	return bows, dictionary

def readObjects(filename):

	# Open the file and read each object/word from every line
	file = open(filename, 'r')
	objects = np.genfromtxt(file, dtype='str')
	file.close()

	return objects