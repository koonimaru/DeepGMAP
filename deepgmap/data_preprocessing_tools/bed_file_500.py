WINDOW_SIZE=200
genome_file="/home/fast/onimaru/data/genome_fasta/mm10.genome"
#with open(genome_file, 'r') as fin, open('/home/fast/onimaru/data/genome_fasta/hg38_1000_altwindow.bed', 'w') as fout1, open('/home/fast/onimaru/data/genome_fasta/hg38_1000_.bed', 'w') as fout2:
with open(genome_file, 'r') as fin, open('/home/fast/onimaru/data/genome_fasta/mm10_200_single_'+str(WINDOW_SIZE)+'.bed', 'w') as fout1:
    
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
            #if i*WINDOW_SIZE+WINDOW_SIZE+WINDOW_SIZE/2<=chrom_size:
                #fout1.write(str(chrom)+'\t'+str(i*WINDOW_SIZE+WINDOW_SIZE/2)+'\t'+str(i*WINDOW_SIZE+WINDOW_SIZE+WINDOW_SIZE/2)+'\n')
            #else:
                #break

