
import sys
import subprocess as sp
import networkx as nx
import os
from itertools import combinations
import glob
from matplotlib import pyplot as plt
import numpy as np
from matplotlib_venn import venn3, venn3_circles

file_list=sorted(glob.glob('/home/fast2/onimaru/DeepGMAP-dev/data/predictions/quick_benchmark/bed_comp_50_es/*'))

file_combination=[]
node_list=[]
peak_counts={}
path_sep=os.path.sep

peak_count={}
peak_count_dict={}
for i in file_list:
    with open(i, 'r') as j:
        peak_count=len(j.readlines())
    
    file_name=i.split(path_sep)
    file_name=file_name[-1].split('.')
    node1=file_name[0]
    peak_counts[node1]=peak_count
    node_list.append(node1)

ABout=open('./intersectAB.bed', 'w')
sp.check_call(["bedtools", "intersect","-u","-F","1.0","-f","1.0", "-a", str(file_list[0]), "-b", str(file_list[1])], stdout=ABout)
ABout.close()
fAB=open('./intersectAB.bed', 'r')
AB=len(fAB.readlines())
fAB.close()
#print AB, peak_counts[node_list[0]]

ABout_=open('./intersectAB_.bed', 'w')
sp.check_call(["bedtools", "intersect","-u","-F","1.0","-f","1.0", "-a", str(file_list[1]), "-b", str(file_list[0])], stdout=ABout_)
ABout_.close()
fAB_=open('./intersectAB_.bed', 'r')
AB_=len(fAB_.readlines())
fAB_.close()
#print AB_, peak_counts[node_list[1]]

if AB>AB_:
    AB=AB_

ACout=open('./intersectAC.bed', 'w')
sp.check_call(["bedtools", "intersect","-u","-F","1.0","-f","1.0", "-a", str(file_list[0]), "-b", str(file_list[2])], stdout=ACout)
ACout.close()
fAC=open('intersectAC.bed', 'r')
AC=len(fAC.readlines())
fAC.close()
#print AC, peak_counts[node_list[2]]

ACout_=open('./intersectAC_.bed', 'w')
sp.check_call(["bedtools", "intersect","-u","-F","1.0","-f","1.0", "-a", str(file_list[2]), "-b", str(file_list[0])], stdout=ACout_)
ACout_.close()
fAC_=open('intersectAC_.bed', 'r')
AC_=len(fAC_.readlines())
fAC_.close()
#print AC_

if AC>AC_:
    AC=AC_

BCout=open('./intersectBC.bed', 'w')
sp.check_call(["bedtools", "intersect","-u","-F","1.0","-f","1.0", "-a", str(file_list[2]), "-b", str(file_list[1])], stdout=BCout)
BCout.close()
fBC=open('intersectBC.bed', 'r')
BC=len(fBC.readlines())
fBC.close()
#print BC

BCout_=open('./intersectBC_.bed', 'w')
sp.check_call(["bedtools", "intersect","-u","-F","1.0","-f","1.0", "-a", str(file_list[1]), "-b", str(file_list[2])], stdout=BCout_)
BCout_.close()
fBC_=open('intersectBC_.bed', 'r')
BC_=len(fBC_.readlines())
fBC_.close()
#print BC_

if BC>BC_:
    BC=BC_

ABCout=open('./intersectABC.bed', 'w')
sp.check_call(["bedtools", "intersect","-u","-F","1.0","-f","1.0", "-a", 'intersectAB.bed', "-b", str(file_list[2])],stdout=ABCout)
ABCout.close()
fABC=open('intersectABC.bed', 'r')
ABC=len(fABC.readlines())
fABC.close()

ABCout_=open('./intersectABC_.bed', 'w')
sp.check_call(["bedtools", "intersect","-u","-F","1.0","-f","1.0", "-b", 'intersectAB.bed', "-a", str(file_list[2])],stdout=ABCout_)
ABCout_.close()
fABC_=open('intersectABC_.bed', 'r')
ABC_=len(fABC_.readlines())
fABC_.close()

if ABC>ABC_:
    ABC=ABC_

Abc=peak_counts[node_list[0]]-AB-AC+ABC
ABc=AB-ABC
AbC=AC-ABC

aBc=peak_counts[node_list[1]]-AB-BC+ABC
aBC=BC-ABC

abC=peak_counts[node_list[2]]-AC-BC+ABC

plt.figure(figsize=(4,4))
v = venn3(subsets=(Abc, aBc, ABc, abC, AbC, aBC, ABC), set_labels = (node_list[0], node_list[1], node_list[2]))
v.get_patch_by_id('100').set_alpha(1.0)
plt.title("Venn diagram")
plt.show()
