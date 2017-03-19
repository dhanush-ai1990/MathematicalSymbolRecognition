#Test for understanding RBM

import numpy as np
from sklearn.neural_network import BernoulliRBM
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
print X
model = BernoulliRBM(n_components=50)
X_new = model.fit_transform(X)

print model.components_
print X_new


