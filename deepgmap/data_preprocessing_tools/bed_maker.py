with open('/home/fast/onimaru/data/mm10.genome', 'r') as fin, open('/home/fast/onimaru/data/mm10_1000_altwindow.bed', 'w') as fout:
    
    for line in fin:
        line=line.split()
        chrom=line[0]
        chrom_size=int(line[1])
        divide_num=chrom_size/1000
        for i in range(divide_num):
            if  i*1000+500>chrom_size:
                break
            if i>=1:
                fout.write(str(chrom)+'\t'+str(i*1000-500)+'\t'+str(i*1000+500)+'\n')