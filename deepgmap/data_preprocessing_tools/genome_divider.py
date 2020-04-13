import sys
import math
import os
import subprocess as sp


def genome_divider(genome_fasta, genome_file, WINDOW_SIZE, outname):
    OUTDIR=os.path.splitext(genome_file)[0]+'_'+str(WINDOW_SIZE)
    try:
        os.makedirs(OUTDIR)
    except OSError as err:
        print("OS error: {0}".format(err))
    outbed=OUTDIR+'/genome.bed'
    outfasta=OUTDIR+'/genome.fa'
    #WINDOW_SIZE=1000
    #genome_file="/home/fast/onimaru/lamprey/LetJap1.0.1.genome"
    #with open(genome_file, 'r') as fin, open('/home/fast/onimaru/data/genome_fasta/hg38_1000_altwindow.bed', 'w') as fout1, open('/home/fast/onimaru/data/genome_fasta/hg38_1000_.bed', 'w') as fout2:
    with open(genome_file, 'r') as fin, open(outbed, 'w') as fout1:
        
        for line in fin:
            line=line.split()
            chrom=line[0]
            chrom_size=int(line[1])
            divide_num=chrom_size/WINDOW_SIZE
            #divide_num=chrom_size/WINDOW_SIZE-4
            for i in range(divide_num):
                
                #if i>=2:
                
                if i*WINDOW_SIZE+WINDOW_SIZE<=chrom_size:
                    fout1.write(str(chrom)+'\t'+str(i*WINDOW_SIZE)+'\t'+str(i*WINDOW_SIZE+WINDOW_SIZE)+'\n')
                else:
                    break
                if i*WINDOW_SIZE+WINDOW_SIZE+WINDOW_SIZE/2<=chrom_size:
                    fout1.write(str(chrom)+'\t'+str(i*WINDOW_SIZE+WINDOW_SIZE/2)+'\t'+str(i*WINDOW_SIZE+WINDOW_SIZE+WINDOW_SIZE/2)+'\n')
                else:
                    break
    try:
        sp.call(["bedtools", "getfasta","-fi",genome_fasta,"-bed",outbed, "-fo", outfasta])
    except OSError as e:
        print(e)
        sys.exit(1)
    
    print(outbed+" and "+outfasta+' were successfully generated.')

def genome_divider2(genome_fasta, genome_file, WINDOW_SIZE, outname, stride=None):
    
    
    if outname is not None:
        OUTDIR=outname
    elif stride is not None:
        OUTDIR=os.path.splitext(genome_file)[0]+'_window'+str(WINDOW_SIZE)+'_stride'+str(stride)
    else:
        OUTDIR=os.path.splitext(genome_file)[0]+'_'+str(WINDOW_SIZE)
    try:
        os.makedirs(OUTDIR)
    except OSError as err:
        #print("OS error: {0}".format(err))
        sys.exit(err)
    
    outbed=OUTDIR+'/genome.bed'
    outfasta=OUTDIR+'/genome.fa'
    """
    if outname is not None:
        outbed=outname+'.bed'
        outfasta=outname+'.fa'
    elif stride is not None:
        outbed=os.path.splitext(genome_file)[0]+'_window'+str(WINDOW_SIZE)+'_stride'+str(stride)+'.bed'
        outfasta=os.path.splitext(genome_file)[0]+'_window'+str(WINDOW_SIZE)+'_stride'+str(stride)+'.fa'
    else:
        outbed=os.path.splitext(genome_file)[0]+'_'+str(WINDOW_SIZE)+'.bed'
        outfasta=os.path.splitext(genome_file)[0]+'_'+str(WINDOW_SIZE)+'.fa'
    """
    #if stride==None:
        #stride=WINDOW_SIZE/2
    #adding=WINDOW_SIZE/stride
    
    #WINDOW_SIZE=1000
    #genome_file="/home/fast/onimaru/lamprey/LetJap1.0.1.genome"
    #with open(genome_file, 'r') as fin, open('/home/fast/onimaru/data/genome_fasta/hg38_1000_altwindow.bed', 'w') as fout1, open('/home/fast/onimaru/data/genome_fasta/hg38_1000_.bed', 'w') as fout2:
    with open(genome_file, 'r') as fin, open(outbed, 'w') as fout1:
        
        for line in fin:
            line=line.split()
            chrom=line[0]
            chrom_size=int(line[1])
            #divide_num=chrom_size/WINDOW_SIZE
            
            #divide_num=chrom_size/WINDOW_SIZE-4
            i=0
            while WINDOW_SIZE+stride*i<=chrom_size:
                fout1.write(str(chrom)+'\t'+str(stride*i)+'\t'+str(WINDOW_SIZE+stride*i)+'\n')
                i+=1
            
    
    try:
        stdout_file=open(outbed+"_tmp", "w")
        sp.check_call(["bedtools", "sort", "-i",outbed], stdout=stdout_file)
        stdout_file.close()
        
    except sp.CalledProcessError as e:
        print("something wrong with bedtools")
        sys.exit(e)
    if os.path.exists(outbed) and os.path.exists(outbed+"_tmp"):
        os.remove(outbed)
        os.rename(outbed+"_tmp", outbed)
    else:
        sys.exit("bed file was not created.")
        
    try:
        sp.check_call(["bedtools", "getfasta","-fi",genome_fasta,"-bed",outbed, "-fo", outfasta])
    except sp.CalledProcessError as e:
        print("something wrong with bedtools")
        sys.exit(e)
    
    print(outbed+" and "+outfasta+' were successfully generated.')

def genome_file_maker(genome_fasta, genome_file):

    length_list=[]
    
    with open(genome_fasta, 'r') as fin, open(genome_file, 'w') as fout:
        seq=0
        chrom_name=''
        for line in fin:
            
            if '>' in line:
                
                if not seq==0:
                    length_list.append(seq)
                    #if not "_" in chrom_name and not "M" in chrom_name:
                    fout.write(str(chrom_name)+'\t'+str(seq)+'\n')
                line=line.split()
                chrom_name=line[0].strip('>')
                seq=0
            else:
                line1=line.strip("\n")
                seq+=len(line1)
        #if len(chrom_name)==3 and not "M" in chrom_name:
        fout.write(str(chrom_name)+'\t'+str(seq)+'\n')
        
def run(args):
    genome_fasta=args.genome_fasta
    windowsize=args.windowsize
    genome_file=os.path.splitext(genome_fasta)[0]+'.genome'
    outname=args.outname
    stride=args.stride
    if not os.path.isfile(genome_file):
        print("generating genome file.")
        genome_file_maker(genome_fasta,genome_file)
    else:
        print("using a pre-existing genome file: "+genome_file)
    genome_divider2(genome_fasta, genome_file, windowsize, outname, stride=stride)
