
infile="/home/fast/onimaru/encode/mm10_dnase-seq_subset/*_summits.bed"
outfile="/home/fast/onimaru/data/CTCF/hiPS_CTCF_peaks.narrowPeak_600.bed"
with open(infile, 'r') as fin, open(outfile, 'w') as fout:
    
    for line in fin:
        
        line=line.split()
        chrom=line[0]
        if not chrom.startswith('chrM') and not '_' in chrom:
            start=int(line[1])
            end=int(line[2])
            mid_p=(start+end)/2
            new_start=mid_p-300
            new_end=mid_p+300
            fout.write(str(chrom)+"\t"+str(new_start)+"\t"+str(new_end)+"\n")
