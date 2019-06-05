import random
import numpy as np

def DNA(length, percentA, percentG, percentC, percentT, percentN):
    a = int(percentA*100)
    g = int(percentG*100)
    c = int(percentC*100)
    t=int(percentT*100)
    n=100-(a+g+c+t)
    dnachoice=''
    i=0
    for i in range(a):
        dnachoice+='A'
    for i in range(g):
        dnachoice+='G'
    i=0
    for i in range(c):
        dnachoice+='C'
    i=0
    for i in range(t):
        dnachoice+='T'
    i=0
    for i in range(n):
        dnachoice+='N'
        
    
    return ''.join(random.choice(str(dnachoice)) for _ in range(length))

def statistics(file):
    lengthdist=[]
    for line in file:
        line=line.split()
        lengthdist.append(int(line[2])-int(line[1]))

    return lengthdist

def AGCTcontent(file2):
    #input_file = open('NC_005213.ffn', 'r') 
    #output_file = open('nucleotide_counts.tsv','w') 
    #output_file.write('Gene\tA\tC\tG\tT\tLength\tCG%\n')
    A_count, C_count, G_count, T_count,N_count, length=0,0,0,0,0,0
    from Bio import SeqIO
    
    for cur_record in SeqIO.parse(file2, "fasta") :
#count nucleotides in this record...
        gene_name = cur_record.name 
        A_count += (cur_record.seq.count('A') +cur_record.seq.count('a'))
        C_count += (cur_record.seq.count('C') +cur_record.seq.count('c'))
        G_count += (cur_record.seq.count('G') +cur_record.seq.count('g'))
        T_count += (cur_record.seq.count('T') +cur_record.seq.count('t'))
        N_count += (cur_record.seq.count('N') +cur_record.seq.count('n'))
        length += len(cur_record.seq)
    A_percent=float(A_count)/float(length)
    G_percent=float(G_count)/float(length)
    C_percent=float(C_count)/float(length)
    T_percent=float(T_count)/float(length)
    N_percent=float(N_count)/float(length)
    #print A_percent, G_percent, C_percent, T_percent, N_percent
    return A_percent, G_percent, C_percent, T_percent, N_percent

with open('/media/koh/HD-PCFU3/mouse/various_dnase_data/all_peak_75cutoff_sorted_merge.bed', 'r') as f1, open('/media/koh/HD-PCFU3/mouse/various_dnase_data/all_peak_75cutoff_sorted_merge.fa', 'r') as f2:
    seq_distribution=statistics(f1)
    percentA, percentG, percentC, percentT, percentN=AGCTcontent(f2)
    output=open('/media/koh/HD-PCFU3/mouse/random_seq/random_for_multidnase.fa', 'w')
    i=0
    for i in range(len(seq_distribution)):
        output.write('>random'+str(i)+'\n'+DNA(seq_distribution[i], percentA, percentG, percentC, percentT, percentN)+'\n')
    output.close()

    