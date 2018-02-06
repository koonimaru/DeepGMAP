bedfile="/home/fast/onimaru/data/genome_fasta/hg38_500.bed"

genome_file="/home/fast/onimaru/data/genome_fasta/hg38.genome"

size_dict={}

with open(genome_file, 'r') as fin:
    for line in fin:
        line=line.split()
        size_dict[line[0]]=int(line[1])

import random
with open(bedfile, 'r') as fin, open('/home/fast/onimaru/data/genome_fasta/hg38_500_rand_250_3times.bed', 'w') as fout:
    for line in fin:
        line=line.split()
        assert len(line)==3
        start=int(line[1])
        end=int(line[2])
        for i in range(3):
            r=random.randint(0,249)
            new_start=start+r
            if new_start+500<size_dict[line[0]]:
                fout.write(line[0]+"\t"+str(new_start)+"\t"+str(new_start+500)+"\n")
            r=random.randint(0,249)
            
            new_start=start-r

            if new_start>0 and new_start+500<size_dict[line[0]]:
                fout.write(line[0]+"\t"+str(new_start)+"\t"+str(new_start+500)+"\n")
        