


def dataset_checker(input1,input2, output1,output2):
    with open(input1, 'r') as fin1,open(input2, 'r') as fin2, open(output1, 'w') as fout1,open(output2, 'w') as fout2:
        
        for line in fin1:
            if '>' in line:
                position=line
                fin2_line=fin2.readline()
            else:
                sequence=line
                N_percent=float(sequence.count('N'))/len(sequence)
                if N_percent<=0.80:
                    fout1.write(str(position)+str(sequence))
                    fout2.write(fin2_line)
if __name__ == '__main__':
    dataset_checker('/home/fast/onimaru/data/mm10_1000_mrg_srt.fa', 
                    '/home/fast/onimaru/data/mm10_1000_mrg_srt.bedadipo_mrg.labeled',
                    '/home/fast/onimaru/data/mm10_1000_mrg_srt_non.fa',
                    '/home/fast/onimaru/data/mm10_1000_mrg_srt.bedadipo_mrg_noN.labeled')