
with open("/home/fast/onimaru/data/mm10_1000_mrg_srt.bed_OCD.labeled", "r") as fin:
    with open("/home/fast/onimaru/data/mm10_1000_mrg_srt.bed_OCD.bed","w") as fout:
        for line in fin:
            a=line.split()
            b=map(int, a[3:])
            if sum(b)>0:
                fout.write(str(a[0])+"\t"+str(a[1])+"\t"+str(a[2])+"\n")