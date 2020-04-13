"""
##gff-version 3
chr2    fimo    nucleotide_motif    5714959    5714967    56.5    +    .    Name=kernel_0_chr2+;Alias=;ID=kernel_0-1-chr2;pvalue=2.23e-06;qvalue= 1;sequence=CGCCTTCGC;
chr2    fimo    nucleotide_motif    10439990    10439998    56.5    +    .    Name=kernel_0_chr2+;Alias=;ID=kernel_0-2-chr2;pvalue=2.23e-06;qvalue= 1;sequence=CGCCTTCGC;
chr2    fimo    nucleotide_motif    13793526    13793534    56.5    +    .    Name=kernel_0_chr2+;Alias=;ID=kernel_0-3-chr2;pvalue=2.23e-06;qvalue= 1;sequence=cgccttcgc;
chr2    fimo    nucleotide_motif    17940241    17940249    56.5    +    .    Name=kernel_0_chr2+;Alias=;ID=kernel_0-4-chr2;pvalue=2.23e-06;qvalue= 1;sequence=CGCCTTCGC;
chr2    fimo    nucleotide_motif    18672533    18672541    56.5    +    .    Name=kernel_0_chr2+;Alias=;ID=kernel_0-5-chr2;pvalue=2.23e-06;qvalue= 1;sequence=CGCCTTCGC;
chr2    fimo    nucleotide_motif    21064760    21064768    56.5    +    .    Name=kernel_0_chr2+;Alias=;ID=kernel_0-6-chr2;pvalue=2.23e-06;qvalue= 1;sequence=CGCCTTCGC;
chr2    fimo    nucleotide_motif    28545836    28545844    56.5    +    .    Name=kernel_0_chr2+;Alias=;ID=kernel_0-7-chr2;pvalue=2.23e-06;qvalue= 1;sequence=CGCCTTCGC;
"""

"""
browser position 
track name="kernels" description="kernel distribution visualization" visibility=2 itemRgb="On"
chr7    127471196  127472363  Pos1  0  +  127471196  127472363  255,0,0
chr7    127472363  127473530  Pos2  0  +  127472363  127473530  255,0,0
chr7    127473530  127474697  Pos3  0  +  127473530  127474697  255,0,0
chr7    127474697  127475864  Pos4  0  +  127474697  127475864  255,0,0
chr7    127475864  127477031  Neg1  0  -  127475864  127477031  0,0,255
chr7    127477031  127478198  Neg2  0  -  127477031  127478198  0,0,255
chr7    127478198  127479365  Neg3  0  -  127478198  127479365  0,0,255
chr7    127479365  127480532  Pos5  0  +  127479365  127480532  255,0,0
chr7    127480532  127481699  Neg4  0  -  127480532  127481699  0,0,255
"""

from matplotlib import pyplot as plt
import numpy as np
import os


cmap = plt.get_cmap('nipy_spectral')
colors = np.array([cmap(i) for i in np.linspace(0, 1, 320)])

colors=(255*colors).astype(int)

gff="/home/fast/onimaru/deepgmap/data/outputs/conv4frss_trained_variables_Fri_May_11_075425_2018_kernels/fimo_out/fimo.gff"
gff="/home/fast2/onimaru/DeepGMAP-dev/data/outputs/conv4frss_Mon_Feb_25_092345_2019_trained_variables_kernels/fimo_out/fimo.gff"
bed=os.path.splitext(gff)[0]+".bed"

with open (gff, 'r') as fin, open(bed, 'w') as fout:
    fout.write('track name="kernels" description="kernel distribution visualization" visibility=2 itemRgb="On"\n')
    for line in fin:
        line=line.split("\t")
        if len(line)==9:
            chr=line[0]
            start=line[3]
            end=line[4]
            #score=line[5]
            orientation=line[6]
            subline=line[-1].split(';')
            for subs in subline:
                if subs.startswith("Name"):
                    subs=subs.split("=")[1].split("_")
                    name=subs[0]+"_"+subs[1]           
                    name_num=int(subs[1])
                elif subs.startswith("pvalue"):
                    subs=-np.log10(float(subs.split("=")[1]))*100
                    score=str(subs)
            _color=",".join(map(str, colors[name_num][:3]))
            fout.write("\t".join([chr, start,end,name,score,orientation,start,end, _color])+"\n")

        