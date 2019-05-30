from deepgmap.misc.small_tools import is_number
import matplotlib.pyplot as plt
import numpy as np

f="/home/fast2/onimaru/DeepGMAP-dev/data/misc/AUPRC_ctcf_boxplot_14jun2018.txt"
#f="/home/fast2/onimaru/DeepGMAP-dev/data/misc/AUPRC_dnase_boxplot_31may2018.txt"
data_list=[]
sample_list=[]

with open(f, 'r') as fin:
    for line in fin:
        data_tmp=[]
        line=line.split()
        if len(line)==0:
            break
        sample_list.append(line[0])
        
        for l in line[1:]:
            if is_number(l) and not l=="nan":
                data_tmp.append(float(l))
        data_list.append(data_tmp)

#print sample_list
#print data_list[-1]
fig, ax = plt.subplots()
font = {'family' : 'Sans',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
ax.boxplot(data_list, labels=sample_list, bootstrap=1000, sym='.')
plt.xticks(rotation='vertical')
ax.grid(True)
"""
ig, ax = plt.subplots()
font = {'family' : 'Sans',
        'weight' : 'normal',
        'size'   : 12}
plt.rc('font', **font)
ax.boxplot(data_list[3:6], labels=sample_list[3:6], bootstrap=1000, sym='.')
"""
#ticks=np.linspace(0, 11, 22, endpoint=False)
#ax.set_yticks(ticks)
#ax.set_x
#locs, labels=plt.xticks()
#plt.xticks(np.arange(len(sample_list)), sample_list)
plt.xticks(rotation='vertical')
ax.grid(True)
"""k=0
for i in [data1, data2]:
    y=i
    x = np.random.normal(k+1, 0.04, len(y))
    plt.plot(x, y,marker="o",linestyle="None")
    k+=1"""
"""
for line in bp_dict['medians']:
    # get position data for median line
    print line.get_xydata()
    x, y = line.get_xydata()[1] # top of median line
    # overlay median value
    plt.text(x+0.15, y-0.1, round(y,2),
         horizontalalignment='center') # draw above, centered"""

import scipy.stats as stats
sub_data_list1=data_list
i=0
pair_set=set()
test_dict={}
for i in range(len(sub_data_list1)):
    for j in range(len(sub_data_list1)):
         
        if not i==j and not str(i)+"-"+str(j) in pair_set:
            #test=stats.ttest_ind(sub_data_list1[i],sub_data_list1[j])
            test=stats.mannwhitneyu(sub_data_list1[i],sub_data_list1[j],alternative="two-sided")
            test_dict[str(i)+"-"+str(j)]=test
            pair_set.add(str(i)+"-"+str(j))
            pair_set.add(str(j)+"-"+str(i))
#print test_dict


plt.show()