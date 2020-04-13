import re

file_name="/home/slow/onimaru/1000genome/HG00119.fa"
i=0
chromosome_list={}
seq=[]
seq_list=[]

with open(file_name,'r') as fin:
    WRITE=False
    for line in fin:
        if line.startswith('>') and "dna:chromosome" in line:
            line=line.split()[0]
            a=line.strip('>')
            a='chr'+str(a)
            seq=[]
            chromosome_list[a]=seq
            seq_list.append(a)
            print(a)
            WRITE=True


        elif line.startswith('>') and "GL" in line:
            WRITE=False
                         
        elif WRITE:
            line1=re.sub(r'\<.*?\>', '', line)
            line1=re.sub(r'\<.*?\n', '', line1)
            line1=re.sub(r'.*?\>', '', line1)
            #line1=line1.strip("\n")
            chromosome_list[a].append(line1.strip('\n'))

with open(file_name+'.ed','w') as fout:
    for k in seq_list:
        #print k
        fout.write(">"+str(k)+"\n")
        for i in chromosome_list[k]:
            fout.write(str(i))
        fout.write("\n")