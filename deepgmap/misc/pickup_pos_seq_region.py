
labeled_file="/home/fast/onimaru/data/CTCF/mm10_CTCF_narrowPeak_mapq/picard_mm10_1000.bed.labeled"

with open(labeled_file, "r") as fin, open(labeled_file.split('.')[0]+"_positive_region.bed", 'w') as fo:
    for line in fin:
        if not line.startswith("#"):
            line1=line.split()
            a=map(int, line1[3:])
            if sum(a) >0:
                fo.write(line)
                
            