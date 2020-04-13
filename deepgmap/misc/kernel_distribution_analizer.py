import numpy as np
#import cPickle
#import gzip
#from sklearn.cluster import DBSCAN
#from sklearn import metrics
import matplotlib.pyplot as plt
#import scipy
import pylab
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as spd
from sklearn.decomposition import PCA, IncrementalPCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import os
from mpl_toolkits.mplot3d import Axes3D
import glob as gl

fname='/home/fast/onimaru/deepgmap/data/outputs/conv4frss_trained_variables_Fri_May_11_075425_2018_kernels/fimo_out/kernels_*_summits_1000.bed'

flist=gl.glob(fname)

pos_dict_dict={}
data_array=[]
mycolors=[]
for f in flist:
    pos_dict={}
    h,t=os.path.split(f)
    t=t.split('_')
    t=t[2]
    print(t)
    with open(f, 'r') as fin:
        for line in fin:
            line=line.split()
            if float(line[7])>=500:
                pos="\t".join(line[:3])
                kernel_num=int(line[6].split("_")[1])
                if not pos in pos_dict:
                    pos_dict[pos]=np.zeros([320], np.float64)
                pos_dict[pos][kernel_num]+=1.0   
    
    pos_dict_dict[t]=pos_dict
    
sample_class=[]
i=0
for k, v in pos_dict_dict.items():
    sample_class.append(k)
    rgb=np.zeros([3], np.float64)
    #rgb[3]=0.5
    if not k=="common":
        rgb[i]=1.0
        i+=1
    for _k,_v in v.items():
        data_array.append(_v)
        mycolors.append(rgb)
    
print(sample_class)
X = np.array(data_array, np.float64)
saving_dir_prefix=fname.split('*')[0]

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
#fig.savefig('/media/koh/HD-PCFU3/mouse/filter_1_clustering.png')


plt.savefig(saving_dir_prefix+'_heat_map.pdf', format='pdf')
"""
tsne = TSNE(n_jobs=16,perplexity = 20.000000,  n_iter=10000)
#X_pca2=np.array(X_pca2, np.float64)
X_tsne = tsne.fit_transform(X)

fig2 = pylab.figure(figsize=(8,8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
             lw=2,s=0.5, c=mycolors)
plt.savefig(saving_dir_prefix+'_tSNE.pdf', format='pdf')


#plt.show()
import pandas as pd
import seaborn as sns
sns.set_style("white")
#df = sns.load_dataset('iris')
 
#my_dpi=96
#plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)
 
# Keep the 'specie' column appart + make it numeric for coloring
#df['species']=pd.Categorical(df['species'])
#my_color=df['species'].cat.codes
#df = df.drop('species', 1)

# Run The PCA
pca = PCA(n_components=3)
pca.fit(X)
 
# Store results of PCA in a data frame
result=pd.DataFrame(pca.transform(X), columns=['PCA%i' % i for i in range(3)])
 
# Plot initialisation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=mycolors, s=10)
 
# make simple, bare axis lines through space:
xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')
 
# label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA on the iris data set")
#plt.show()

plt.show()"""