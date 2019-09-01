# pyJSMF

Python implementation of Joint Stochastic Matrix Factorization (JSMF) for the Rectified Anchor Word (RAW) algorithm.

Co-occurrence information is powerful statistics that can model various discrete objects by their joint instances with other objects. Transforming unsupervised problems of learning low-dimensional geometry into provable decompositions of co-occurrence information, spectral inference provides fast algorithms and optimality guarantees for non-linear dimensionality reduction and latent topic analysis. Spectral approaches reduce the dependence on the original training examples, thereby producing substantial gain in efficiency, but at costs:

- The algorithms perform poorly on real data that does not necessarily follow underlying models.
- Users can no longer infer information about individual examples, which is often important for real-world applications.
- Model complexity rapidly grows as the number of objects increases, requiring a careful curation of the vocabulary.

The first issue is called model-data mismatch, which is a fundamental problem common in every spectral inference method for latent variable models. As real data never follows any particular computational model, this issue must be addressed for practicality of the spectral inference beyond synthetic settings.

The rectification paradigm in this code provides a neat solution to handle model-data mismatch not making more complex models but transforming data to a point in the space of ideal posterior. 


Examples:

1. Create full co-occurrence statistics C from bag-of-words
```
from pyJSMF.file import createC
C, D1, D2, dictionary = createC("dataset/real_bows/docword.nips.txt", "dataset/real_bows/vocab.nips.txt", "dataset/standard.stops", N=5000, min_objects=3, min_tokens=5, output_filename="example")
```

2. Factorize full C matrix for 5 topics after rectification via Alternating Projection with active-set method for optimizing B
```
from pyJSMF.factorize import factorizeC
S, B, A, Btilde, Cbar, C_rowSums, diagR, C = factorizeC(C, K=5, rectifier='AP', optimizer='activeSet')
```

3. Factorize full C matrix for 10 topics after rectification via Douglas-Rachford iterations with admmDR method for optimizing B
```
S, B, A, Btilde, Cbar, C_rowSums, diagR, C = factorizeC(C, K=10, rectifier='DR', optimizer='admmDR')
```

4. Print out 7 top words from each topic using the dictionary in example.dict
```
from pyJSMF.evaluation import generateTopWords
generateTopWords(S, B, 7, "example.dict")
```

5. Quantitatively evaluate with various metrics
```
from pyJSMF.evaluation import evaluateClusters
title, value, stdev = evaluateClusters(B, A, S, Btilde, Cbar, C_rowSums, C, D1, D2)
```

# Reference
[Background paper]
- https://www.aclweb.org/anthology/D14-1138

[Related work]
- https://arxiv.org/abs/1212.4777
- https://arxiv.org/abs/1204.1956
