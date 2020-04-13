
infile="/home/fast/onimaru/data/CTCF/hg38_200_no_hiPS_CTCF.bed"
outfile="/home/fast/onimaru/data/CTCF/hg38_200_no_hiPS_CTCF_pm400.bed"
with open(infile, 'r') as fin, open(outfile, 'w') as fout:
    
    for line in fin:
        line=line.split()
        chrom=line[0]
        start=int(line[1])
        end=int(line[2])
        new_start=start-400
        new_end=end+400
        fout.write(str(chrom)+"\t"+str(new_start)+"\t"+str(new_end)+"\n")
