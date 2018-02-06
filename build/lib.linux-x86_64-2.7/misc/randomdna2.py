import random



input_file='/home/fast/onimaru/data/various_dnase_data/mm10_1000_dnase_region_75co.fa'
temp_file='/home/fast/onimaru/data/random_seq/random_shuffle2_for_mm10_1000_dnase_region_75co.fa'
output_file='/media/koh/HD-PCFU3/mouse/random_seq/random_shuffle2_for_multidnase_no_chr1_2_noN.fa'
with open(input_file, 'r') as f1, open(temp_file, 'w') as f2:
    for line in f1:
        if '>' in line:
            f2.write(str(line))
        elif not line=='' or not line=='\n':
            randomized=''
            select_data=random.randint(1,100)
            shuffling_module=2
            #if select_data<=50:
                #shuffling_module=3
            line=line.strip('\n')
            index_new=(len(line))/shuffling_module
            a = range(index_new)
            index_random = random.sample(a, len(a))
            index_random_iter=iter(index_random)
            for i in range(index_new):
                for k in range(shuffling_module):
                    randomized+=line[index_random[i]*shuffling_module+k]
            f2.write(str(randomized)+'\n')
for i in range(2):
    with open(input_file, 'r') as f1, open(temp_file, 'a') as f2:
        for line in f1:
            if '>' in line:
                f2.write(str(line))
            elif not line=='' or not line=='\n':
                randomized=''
                select_data=random.randint(1,100)
                shuffling_module=2
                #if select_data<=50:
                    #shuffling_module=3
                line=line.strip('\n')
                index_new=(len(line))/shuffling_module
                a = range(index_new)
                index_random = random.sample(a, len(a))
                index_random_iter=iter(index_random)
                for i in range(index_new):
                    for k in range(shuffling_module):
                        randomized+=line[index_random[i]*shuffling_module+k]
                f2.write(str(randomized)+'\n')
