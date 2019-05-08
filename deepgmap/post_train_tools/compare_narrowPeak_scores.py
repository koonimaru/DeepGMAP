import math

"""
chr1    0    1000    .     3.64092565    .     0.00364093    -1    -1    -1
chr1    500    1500    .     3.64092565    .     0.00364093    -1    -1    -1
chr1    1000    2000    .     3.64092565    .     0.00364093    -1    -1    -1
chr1    1500    2500    .     3.64092565    .     0.00364093    -1    -1    -1
"""


ref_data="/home/fast/onimaru/data/prediction/CTCF/network_constructor_deepsea_1d3_Wed_Oct_11_074555_2017.ckpt-13019.narrowPeak"
ind_data="/home/fast/onimaru/data/prediction/CTCF/HG00119_network_constructor_deepsea_1d3_Wed_Oct_11_074555_2017.ckpt-13019.narrowPeak.hg38.narrowPeak"

ref_data_dict={}
ind_data_dicts={}

with open(ref_data,'r') as fin:
    for line in fin:
        line=line.split()
        position=str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])
        score=float(line[4])
        ref_data_dict[position]=score
        
with open(ind_data,'r') as fin:
    for line in fin:
        line_=line.split()
        position=str(line_[0])+'\t'+str(line_[1])+'\t'+str(line_[2])
        score=float(line_[4])
        if position in ref_data_dict:
            score_of_ref=ref_data_dict[position]
            abs_diff=math.fabs(score-score_of_ref)
            
        ref_data_dict[position]=score