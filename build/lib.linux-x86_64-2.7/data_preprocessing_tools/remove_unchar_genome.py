genome="/home/slow/onimaru/data/genome_fasta/hg38.fa"
genome_out="/home/slow/onimaru/data/genome_fasta/hg38_chromosome_only.fa"
with open( genome, 'r') as fin, open(genome_out,'w') as fout:
    WRITE=False
    for line in fin:
        if line.startswith(">"):
            if not "_" in line and not "chrM" in line:
                fout.write(line)
                WRITE=True
                
            else:
                WRITE=False
                
        elif WRITE==True:
            fout.write(line)
        