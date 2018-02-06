import numpy as np
import cPickle
import gzip
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy
import pylab
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
from sklearn.decomposition import PCA, IncrementalPCA

with gzip.open('/media/koh/HD-PCFU3/mouse/variables_4999_Sun_Jan_22_181643_2017.cpickle.gz', 'r') as f1:
    learned_variables=cPickle.load(f1)
    filter1=learned_variables['W_conv1']
    filter1_shape=filter1.shape
    filter1_flattened_array=scipy.zeros((filter1_shape[3], filter1_shape[0]*filter1_shape[1]), np.float32)    
    for i in range(filter1_shape[3]):            
        for k in range(filter1_shape[0]):
            for l in range(filter1_shape[1]):
                a=filter1[k][l][0][i]
                filter1_flattened_array[i][l+k*4]+=a
    
X = filter1_flattened_array
D = spd.pdist(X, 'cosine')
# Compute and plot first dendrogram.
fig = pylab.figure(figsize=(8,8))
ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
Y = sch.linkage(D, method='ward')
Z1 = sch.dendrogram(Y, orientation='left')
ax1.set_xticks([])
ax1.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
#idx2 = Z2['leaves']
X2 = X[idx1]
im = axmatrix.matshow(X2, aspect='auto', origin='lower', cmap=pylab.get_cmap('YlGnBu'))
axmatrix.set_xticks([])
axmatrix.set_yticks([])

# Plot colorbar.
axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
pylab.colorbar(im, cax=axcolor)
fig.savefig('/media/koh/HD-PCFU3/mouse/filter_1_clustering.png')
pylab.show()

"""
import matplotlib.pyplot as mplt


db = DBSCAN(eps=0.3,min_samples=3, algorithm='auto').fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
colors = mplt.get_cmap('Spectral')(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    mplt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    mplt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

mplt.title('Estimated number of clusters: %d' % n_clusters_)
mplt.show()

n_components = 2
ipca = IncrementalPCA(n_components=n_components, batch_size=512, whiten=True)
X_ipca = ipca.fit_transform(X)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)
for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    
    plt.scatter(X_transformed[0], X_transformed[1], lw=2)

    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
        plt.title(title + " of feature map\nMean absolute unsigned error "
                  "%.6f" % err)
    else:
        plt.title(title + " of feature map")
    plt.legend(loc="best", shadow=False, scatterpoints=1)

mplt.show()      """