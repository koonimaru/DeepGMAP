import re
map_file="/home/slow/onimaru/1000genome/hg38_HG00119_1000.outfmt"


with open(map_file, 'r') as fin:
    has_seen=set()
    map_dict={}
    for line in fin:
        line=line.split()
        hg38_coo, indiv_coo=line[1], line[0]
        if not hg38_coo in has_seen:
            has_seen.add(hg38_coo)
            map_dict[indiv_coo]=hg38_coo
            
narrowPeak_prediction="/home/fast/onimaru/data/prediction/HG00119/HG00119_network_constructor_deepsea_1d3_Wed_Oct_11_074555_2017.ckpt-13019.narrowPeak"

with open(narrowPeak_prediction, 'r') as fin, open(narrowPeak_prediction+".hg38.narrowPeak", 'w') as fout:
    for line in fin:
        a=line.split()
        b=str(a[0])+":"+str(a[1])+"-"+str(a[2])
        if b in map_dict:
            new_coo=map_dict[b]
            new_coo=re.findall(r"[\w']+", new_coo)
            fout.write(str(new_coo[0])+"\t"
                       +str(new_coo[1])+"\t"
                       +str(new_coo[2])+"\t"
                       +str(a[3])+"\t"
                       +str(a[4])+"\t"
                       +str(a[5])+"\t"
                       +str(a[6])+"\t"
                       +str(a[7])+"\t"
                       +str(a[8])+"\t"
                       +str(a[9])+"\n")
