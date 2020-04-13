def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
import matplotlib.pyplot as plt
import numpy as np
f="/home/fast/onimaru/deepgmap/data/misc/deepsea_S3.txt"

datadict={}

with open(f, 'r') as fin:
    for line in fin:
        
        line=line.split("\t")
        if len(line)==7:
            #print line[1], line[5]
            if is_number(line[5]):
                if datadict.has_key(line[1]):
                    datadict[line[1]].append(float(line[5]))
                else:
                    datadict[line[1]]=[]
                    datadict[line[1]].append(float(line[5]))

data_list=[]
label_list=[]

for k, v in datadict.items():
    if len(v)>3:
        label_list.append(k)
        data_list.append(v)

median_list=[]

for i in data_list:
    median_list.append(np.median(i))

index_=range(len(label_list))
        
median_list, index_=zip(*sorted(zip(median_list, index_), reverse=True))

label_list[:] = [label_list[i] for i in index_]
data_list[:] = [data_list[i] for i in index_]


fig, ax = plt.subplots()
font = {'family' : 'Sans',
        'weight' : 'normal',
        'size'   : 6}
plt.rc('font', **font)
bp_dict=ax.boxplot(data_list, labels=label_list, bootstrap=1000, sym='.')
#ticks=np.linspace(0, 11, 22, endpoint=False)
#ax.set_yticks(ticks)
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
"""
import scipy.stats as stats

test=stats.ttest_ind(data1,data2)
test2=stats.ttest_ind(data1,data3)
print test, test2
"""
plt.show()