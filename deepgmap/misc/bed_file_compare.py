
import sys
import subprocess as sp
import networkx as nx
import os
from itertools import combinations
import glob

file_list=glob.glob(sys.argv[1])

file_combination=[]
node_list={}
peak_counts={}
path_sep=os.path.sep
G=nx.MultiDiGraph()
G.peak_count={}
peak_count_dict={}
for i in file_list:
    with open(i, 'r') as j:
        peak_count=len(j.readlines())
    
    file_name=i.split(path_sep)
    file_name=file_name[-1].split('.')
    node1=file_name[0]
    G.add_node(node1)
    G.peak_count[node1]=peak_count
    peak_count_dict[node1]=str(node1)+'\n('+ str(peak_count)+')'
    node_list[i]=node1
    
for i in combinations(file_list, 2):
    file_combination.append(i)
    
edgelabels={}

fout=open('/media/koh/HD-PCFU3/mouse/various_dnase_data/bedfiles/testfiles/test.log', 'w')
fout.write("#combination, intersection, overlapping, distance\n")


for i in file_combination:
    
    intersect1_=sp.check_output(["bedtools", "intersect","-u", "-a", str(i[0]), "-b", str(i[1])])
    intersect1=len(intersect1_.split('\n'))
    intersect2_=sp.check_output(["bedtools", "intersect","-u", "-b", str(i[0]), "-a", str(i[1])])
    intersect2=len(intersect2_.split('\n'))
    #distance=sp.check_output(["bedtools", "jaccard", "-a", str(i[0]), "-b", str(i[1])])
    #distance=distance.split('\n')
    #distance=distance[1].split()
    #distance=distance[2]
    overlap1=G.peak_count[node_list[i[0]]]-intersect1
    overlap2=G.peak_count[node_list[i[1]]]-intersect2
    
    proportion1=overlap1/float(G.peak_count[node_list[i[0]]])
    proportion2=overlap2/float(G.peak_count[node_list[i[1]]])
    
    fout.write(str(node_list[i[0]])+'/'+str(node_list[i[1]])+', '+str(intersect1)+'/'+str(intersect2)+', '+str(overlap1)+'/'+str(overlap2)+', '+str(proportion1)+'/'+str(proportion2)+'\n')
    G.add_edge(node_list[i[0]], node_list[i[1]], edge_width=float(proportion1))
    G.add_edge(node_list[i[1]], node_list[i[0]], edge_width=float(proportion2))
    edgelabels[node_list[i[0]], node_list[i[1]]]=str(overlap1)+'/'+str(overlap2)
fout.close()
import matplotlib.pyplot as plt
edgewidth=[]
for (u,v,d) in G.edges(data=True):
    edgewidth.append(d['edge_width'])
    
plt.figure(figsize=(8,8))
# with nodes colored by degree sized by population
pos=nx.spectral_layout(G)
nx.draw_networkx_edges(G,pos,alpha=0.3,width=20, edge_color=edgewidth)
nodesize=[G.peak_count[v]/100 for v in G]
nx.draw_networkx_nodes(G,pos,node_size=nodesize,node_color='w',alpha=1.0,label=nodesize)
nx.draw_networkx_labels(G,pos,labels=peak_count_dict, fontsize=14)
nx.draw_networkx_edge_labels(G,pos,edge_labels=edgelabels, fontsize=12, alpha=0.1, bbox=dict(facecolor='none', edgecolor='none'))
plt.savefig("chess_masters.png",dpi=75)
plt.show()
