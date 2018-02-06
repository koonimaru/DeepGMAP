
"""
genome_file='/home/slow/onimaru/1000genome/HG00119_chr1.genome'

with open('/home/slow/onimaru/1000genome/HG00119.fa.ed','r') as fin, open(genome_file,'w') as fout:
    chrom=None
    length=0
    for line in fin:

        if line.startswith('>'):
            if not length==0:
                #print(str(chrom)+"\t"+str(length)+"\n")
                #if not chrom=="chrM" and not "_" in chrom:
                if chrom=="1":
                    print(str(chrom)+"\t"+str(length))
                    fout.write("chr"+str(chrom)+"\t"+str(length)+"\n")
                chrom=None
                length=0
            
            #chrom=line.strip('>\n')
            chrom=line.split()[0].strip('>')
            print chrom
        else:
            line=line.strip('\n')
            #print length
            length=length+len(line)
    #if not chrom=="chrM" and not "_" in chrom:
        #fout.write(str(chrom)+"\t"+str(length)+"\n")
"""

WINDOW_SIZE=300
genome_file="/home/fast/onimaru/data/genome_fasta/mm10.genome"
#with open(genome_file, 'r') as fin, open('/home/fast/onimaru/data/genome_fasta/hg38_1000_altwindow.bed', 'w') as fout1, open('/home/fast/onimaru/data/genome_fasta/hg38_1000_.bed', 'w') as fout2:
with open(genome_file, 'r') as fin, open('/home/fast/onimaru/data/genome_fasta/mm10_'+str(WINDOW_SIZE)+'_300.bed', 'w') as fout1:
    
    for line in fin:
        line=line.split()
        chrom=line[0]
        chrom_size=int(line[1])
        divide_num=chrom_size/WINDOW_SIZE
        #divide_num=chrom_size/WINDOW_SIZE-4
        for i in range(divide_num):
            
            #if i>=2:
            
            if i*WINDOW_SIZE+WINDOW_SIZE<=chrom_size:
                fout1.write(str(chrom)+'\t'+str(i*WINDOW_SIZE)+'\t'+str(i*WINDOW_SIZE+WINDOW_SIZE)+'\n')
            else:
                break
            if i*WINDOW_SIZE+WINDOW_SIZE+WINDOW_SIZE/2<=chrom_size:
                fout1.write(str(chrom)+'\t'+str(i*WINDOW_SIZE+WINDOW_SIZE/2)+'\t'+str(i*WINDOW_SIZE+WINDOW_SIZE+WINDOW_SIZE/2)+'\n')
            else:
                break

