#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.decomposition import PCA

#%%
# create dataset for regression
X, y = make_regression(n_samples=100, n_features=1, noise=5)
# plot regression dataset

#%%

#%%
round(min(X)[0])
#%%
vecs = np.array(list(zip(map(lambda x: x[0], X), y)))

#%%
# pca to see line of best fit

pca = PCA(n_components=2)
pca.fit(vecs)

x = print(pca.singular_values_)

# plt.scatter(pca.singular_values_)
x


#%%
def line_from_two_points(point_a, point_b):
    # TODO rewrite in a way you understand
    from numpy import ones, vstack
    from numpy.linalg import lstsq
    points = [point_a, point_b]
    x_coords, y_coords = zip(*points)
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    print("Line Solution is y = {m}x + {c}".format(m=m, c=c))
    return (m, c)


#%%
print(pca.explained_variance_ratio_)
pca.components_
# this the "regression" from pca
m, c = line_from_two_points(pca.components_[0], [0, 0])
x = range((int(min(X)[0])), int(max(X)[0]) + 1)

#%%
regr = linear_model.LinearRegression()

# Train the model
regr.fit(X, y)

#%%
plt.plot(x, m * x + c, 'r', label='Fitted line')
plt.plot(X, regr.predict(X), color='orange')
plt.scatter(X, y)
