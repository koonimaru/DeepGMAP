import random
labeled_file="/home/fast/onimaru/data/Chip-seq/three_tfs_hg38_500_rand_250_5times_srt.bed.labeled"
labeled_file_out="/home/fast/onimaru/data/Chip-seq/three_tfs_hg38_500_rand_250_5times_srt_reduced.bed.labeled"
with open(labeled_file, 'r') as fin, open(labeled_file_out ,'w') as fout:
    for line in fin:
        if line.startswith("#"):
            fout.write(line)
        else:
            r=random.random()
            line1=line.split()
            #print line1[3:]
            label_num=sum(map(int, line1[3:]))
            if label_num==0 and r<=0.800:
                continue
            else:
                fout.write(line)
            