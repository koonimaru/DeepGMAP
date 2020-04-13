file_name="/home/slow/onimaru/1000genome/HG00119.fa.ed"
file_out="/home/slow/onimaru/1000genome/HG00119_ed_chr.fa"
with open(file_name, "r") as fin, open(file_out,"w") as fout: 
    for line in fin:
        if line.startswith('>'):
            line=line.split()
            chromo=line[0].strip('>')
            line=">chr"+str(chromo)+"\n"
            fout.write(line)
        else:
            fout.write(line)
