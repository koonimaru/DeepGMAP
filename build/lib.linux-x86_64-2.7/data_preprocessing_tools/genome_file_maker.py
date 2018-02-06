import sys
import math
length_list=[]


with open(sys.argv[1], 'r') as fin, open('./'+sys.argv[2], 'w') as fout:
    seq=0
    chrom_name=''
    for line in fin:
        
        if '>' in line:
            
            if not seq==0:
                length_list.append(seq)
                #if not "_" in chrom_name and not "M" in chrom_name:
                fout.write(str(chrom_name)+'\t'+str(seq)+'\n')
            line=line.split()
            chrom_name=line[0].strip('>')
            seq=0
        else:
            line1=line.strip("\n")
            seq+=len(line1)
    #if len(chrom_name)==3 and not "M" in chrom_name:
    fout.write(str(chrom_name)+'\t'+str(seq)+'\n')