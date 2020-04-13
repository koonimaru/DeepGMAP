

chromosome_name="chr1"
file_name="/home/slow/onimaru/1000genome/HG00119_1000.fa"
with open(file_name, "r") as fin, open(file_name+"_"+chromosome_name, "w") as fout:
    WRITE=False
    for line in fin:
        if line.startswith('>'):
            #line1=line.split()
            a=line.strip('>\n')
            #print a
            if a.startswith(chromosome_name):
                fout.write(line)
                WRITE=True
            else:
                WRITE=False
        
        elif WRITE:
            fout.write(line)
