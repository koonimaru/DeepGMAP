


def dataset_checker(input, output):
    with open(input, 'r') as f_in, open(output, 'w') as f_out:
        for line in f_in:
            if '>' in line:
                position=line
            else:
                sequence=line
                N_percent=float(sequence.count('N'))/len(sequence)
                if N_percent<0.90 and len(sequence)>100:
                    f_out.write(str(position)+str(sequence))
            
if __name__ == '__main__':
  dataset_checker('/home/fast/onimaru/data/CTCF/mm10_no_CTCF.fa', '/home/fast/onimaru/data/CTCF/mm10_no_CTCF_noN.fa')