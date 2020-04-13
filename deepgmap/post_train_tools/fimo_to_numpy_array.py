import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
import subprocess as sp

"""
fimo_format
# motif_id    motif_alt_id    sequence_name    start    stop    strand    score    p-value    q-value    matched_sequence
MA0139.1    CTCF    chr2    231612721    231612739    +    27.1967    4.53e-12    0.00802    TGGCCACCAGGGGGCGCCG
MA0139.1    CTCF    chr16    84970710    84970728    -    27.1311    6.58e-12    0.00802    CGGCCACCAGGGGGCGCCA
MA0139.1    CTCF    chr3    98023412    98023430    -    27.1311    6.58e-12    0.00802    CGGCCACCAGGGGGCGCCA
MA0139.1    CTCF    chr5    137499397    137499415    -    27.1311    6.58e-12    0.00802    CGGCCACCAGGGGGCGCCA
"""



x=[]

y=[]

fimo_file="/home/fast/onimaru/mouse/fimo_out/fimo.txt"
narrowpeak_file="/home/fast/onimaru/mouse/fimo_out/fimo_logq33.narrowPeak"
intersect_file='/home/fast/onimaru/mouse/fimo_out/fimo_1000_logq33.narrowPeak'
bed_file='/home/fast/onimaru/mouse/fimo_out/fimo_1000_logq33.bed'
genome_1000='/home/slow/onimaru/data/genome_fasta/mm10_1000.bed'
prediction_arrray="/home/fast/onimaru/mouse/fimo_out/fimo_prediction_all"
target="all"
logq_threshold=0.33

with open(fimo_file, 'r') as fin:
    with open(narrowpeak_file, 'w') as fout:
        i=0
        for line in fin:
            if not line[0]=="#":
                a=line.split()
                chromo=a[2]
                start=int(a[3])
                end=int(a[4])
                name='fimo_'+str(a[0])+'_'+str(a[1])
                orientation=a[5]
                score=float(a[6])
                logp=-np.log10(float(a[7]))
                logq=-np.log10(float(a[8]))
                if logq>=logq_threshold:
                    fout.write(str(chromo)+"\t"+
                               str(start)+"\t"+
                               str(end)+"\t"+
                               str(name)+"\t"+
                               str(logq*400)+"\t"+
                               str(orientation)+"\t"+
                               str(score)+"\t"+
                               str(logp)+"\t"+
                               str(logq)+"\t"+
                               "-1\n"
                               )
                
                
            i+=1
            if i%10000==0:
                print("reading "+str(i) + "th line of fimo file")

print("converting narrowPeak to 1000 binned peaks")
intersectout=open(intersect_file, 'w')
sp.check_call(["bedtools", "intersect","-F","0.4","-wo", "-a", str(genome_1000), "-b", str(narrowpeak_file)], stdout=intersectout)
intersectout.close()
print("conversion is done")
"""
chr1    10500    11500    chr1    11223    11241    fimo_MA0139.1_CTCF    675.298455578    -    24.4754    8.87289520164    1.68824613894    -1    18
chr1    10500    11500    chr1    11281    11299    fimo_MA0139.1_CTCF    566.267510253    -    22.7377    7.99567862622    1.41566877563    -1    18
chr1    11000    12000    chr1    11223    11241    fimo_MA0139.1_CTCF    675.298455578    -    24.4754    8.87289520164    1.68824613894    -1    18
chr1    11000    12000    chr1    11281    11299    fimo_MA0139.1_CTCF    566.267510253    -    22.7377    7.99567862622    1.41566877563    -1    18
"""




#intersect_file='/home/fast/onimaru/human/fimo_out_1e3/fimo_cutoff_0p33_logq.narrowPeak_test.bed'
fimo_peak_dict={}

if target=="all":
    startswtith="chr"
else:
    startswtith=str(target)+"\t"
    

with open(intersect_file,"r") as fin:
    for line in fin:
        if line.startswith(startswtith):
            a=line.split()
            position=str(a[0])+"\t"+str(a[1])+"\t"+str(a[2])
            logq=float(a[11])
            if not position in fimo_peak_dict:
                fimo_peak_dict[position]=logq
            elif logq>fimo_peak_dict[position]:
                fimo_peak_dict[position]=logq
        

#genome_1000='/home/slow/onimaru/data/genome_fasta/hg38_1000.bed'
qvalue_list=[]
with open(genome_1000, "r") as fin, open(bed_file,'w') as fout:
    for line in fin:
        if line.startswith(startswtith):
            a=line.strip('\n')
            if fimo_peak_dict.has_key(a):
                fout.write(a+"\n")
                qvalue_list.append(fimo_peak_dict[a])
                print(line)
            else:
                qvalue_list.append(0.00)


qvalue_array=np.array(qvalue_list)/np.max(qvalue_list)

np.savez_compressed(prediction_arrray, prediction=qvalue_array)
        







                
"""
x=np.array(x)/np.max(x)

np.savez_compressed("/home/fast/onimaru/human/fimo_out_1e3/fimo_prediction", prediction=x)

# the histogram of the data

plt.subplot(211)
n, bins, patches = plt.hist(x, 50, facecolor='green', alpha=0.5)
plt.yscale('log', nonposy='clip')
#plt.hist(x, 50, facecolor='red', alpha=0.5, cumulative=True)
# add a 'best fit' line

#l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('-Log10(p value)')
#plt.ylabel('Scores')
plt.title('fimo_prediction_dist')
#plt.axis([40, 160, 0, 0.03])
plt.subplot(212)
plt.hist(y, 50, facecolor='blue', alpha=0.5)
plt.grid(True)

plt.show()"""