cimport cython
import sys
import time
import random

@cython.boundscheck(False)
@cython.wraparound(False)
def label_reader(list file_, list chr_to_skip, float reduce):
    cdef list label_position=[]
    label_position_append=label_position.append
    cdef list label_list=[]
    label_list_append=label_list.append
    cdef list line_=[]
    cdef int i=0, value_sum, skipped=0, pos_no=0
    cdef str line, p, skipped_chr=""
    cdef int _name_length
    cdef list value
    cdef float r
    cdef int skipping=0
    #cdef list skipped=[]
    #skipped_append=skipped.append
    for line in file_:
        #print(line)
        if line[0]=="#":
            continue
        
        line_=line.split()
        for _chr in chr_to_skip:
            if line_[0]==_chr:
                if not skipped_chr==line_[0]:
                    sys.stdout.write('skipping '+line_[0]+'\r')
                    sys.stdout.flush()
                    skipped_chr=line_[0]
                    skipping=1
                break
            else:
                skipping=0
        if skipping==1:
            continue
        
        value=map(int, line_[3:])
        value_sum=sum(value)
        r=random.random()
        p="".join([str(line_[0]),':',str(line_[1]),'-',str(line_[2])])
        if value_sum==0 and r<reduce:
            skipped+=1
        else:
            if value_sum>0:
                pos_no+=1
            label_position_append(p)
            label_list_append(value)
        
        
        i+=1
        if i%100000==0:
            
            #sys.exit()
            sys.stdout.write('reading labels %i \r' % (i))
            sys.stdout.flush()
    return label_position, label_list, skipped, pos_no




@cython.boundscheck(False)
@cython.wraparound(False)
def label_reader2(list file_, list chr_to_skip, float reduce):
    cdef list label_position=[]
    label_position_append=label_position.append
    cdef list label_list=[]
    label_list_append=label_list.append
    cdef list line_=[]
    cdef int i=0, value_sum, skipped=0, pos_no=0
    cdef str line, p, skipped_chr=""
    cdef int _name_length
    cdef list value
    cdef float r
    cdef int skipping=0
    #cdef list skipped=[]
    #skipped_append=skipped.append
    #start=time.time()
    for line in file_:
        #print(line)
        if line[0]=="#":
            continue
        
        line_=line.split()
        for _chr in chr_to_skip:
            if line_[0]==_chr:
                skipping=1
                if not skipped_chr==line_[0]:
                    sys.stdout.write('skipping '+line_[0]+'\r')
                    sys.stdout.flush()
                    skipped_chr=line_[0]
                
                break
            else:
                skipping=0
        #print(line_, skipping)
        if skipping==1:
            continue
        
        value=line_[3:]
        value_sum=0
        for v in value:
            if v=="1":
                value_sum=1
                break
                
        #value_sum=sum(value)
        r=random.random()
        p="".join([str(line_[0]),':',str(line_[1]),'-',str(line_[2])])
        if value_sum==0 and r<reduce:
            skipped+=1
        else:
            if value_sum>0:
                pos_no+=1
            label_position_append(p)
            label_list_append(value)
        
        i+=1
        if i%100000==0:
            #print time.time()-start
            #sys.exit()
            sys.stdout.write("reading label "+ str(line))
            sys.stdout.flush()
    return label_position, label_list, skipped, pos_no




def AGCTtoArray2(char *Seq, int seqlen):
    cdef list onehot=[]
    #onehot_append=onehot.append
    cdef char Nuc
    cdef int i
    for i in range(seqlen):
        Nuc=Seq[i]
        #print Nuc
        if Nuc=='A' or Nuc=='a':
            onehot.append((1, 0, 0, 0))
        elif Nuc=='G' or Nuc=='g':
            onehot.append((0, 1, 0, 0))
        elif Nuc=='C' or Nuc=='c':
            onehot.append((0, 0, 1, 0))
        elif Nuc=='T' or Nuc=='t':
            onehot.append((0, 0, 0, 1))
        elif Nuc=='N' or Nuc=='n':
            onehot.append((0, 0, 0, 0))
        else:
            print("sequence contains unreadable characters: "+str(Nuc))
            sys.exit()
    
    return onehot


def AGCTtoArray3(char *Seq, int seqlen):
    cdef list onehot=[]
    onehot_append=onehot.append
    cdef char Nuc
    cdef int i
    for i in range(seqlen):
        Nuc=Seq[i]
        #print Nuc
        if Nuc=='A' or Nuc=='a':
            onehot_append((1, 0, 0, 0))
        elif Nuc=='G' or Nuc=='g':
            onehot_append((0, 1, 0, 0))
        elif Nuc=='C' or Nuc=='c':
            onehot_append((0, 0, 1, 0))
        elif Nuc=='T' or Nuc=='t':
            onehot_append((0, 0, 0, 1))
        elif Nuc=='N' or Nuc=='n':
            onehot_append((0, 0, 0, 0))
        else:
            print("sequence contains unreadable characters: "+str(Nuc))
            sys.exit()
    
    return onehot

def AGCTtoArray5(char *Seq, int seqlen):
    cdef list onehot=[]
    onehot_append=onehot.append
    cdef char Nuc
    cdef int i
    for i in range(seqlen):
        Nuc=Seq[i]
        #print Nuc
        if Nuc=='A' or Nuc=='a':
            onehot_append((1, 0, 0, 0))
        elif Nuc=='G' or Nuc=='g':
            onehot_append((0, 1, 0, 0))
        elif Nuc=='C' or Nuc=='c':
            onehot_append((0, 0, 1, 0))
        elif Nuc=='T' or Nuc=='t':
            onehot_append((0, 0, 0, 1))
        elif Nuc=='N' or Nuc=='n':
            onehot_append((0, 0, 0, 0))
        else:
            print("sequence contains unreadable characters: "+str(Nuc))
            sys.exit()
    
    return onehot

@cython.boundscheck(False)
@cython.wraparound(False)
def AGCTtoArray4(char *Seq, int seqlen):
    cdef list onehot=[]
    onehot_append=onehot.append
    cdef char Nuc
    cdef int i, j=0
    for i in range(seqlen):
        Nuc=Seq[i]
        #print Nuc
        if Nuc=='A' or Nuc=='a':
            onehot_append((1, 0, 0, 0))
        elif Nuc=='G' or Nuc=='g':
            onehot_append((0, 1, 0, 0))
        elif Nuc=='C' or Nuc=='c':
            onehot_append((0, 0, 1, 0))
        elif Nuc=='T' or Nuc=='t':
            onehot_append((0, 0, 0, 1))
        elif Nuc=='N' or Nuc=='n':
            onehot_append((0, 0, 0, 0))
        else:
            onehot_append((0, 0, 0, 0))
            j+=1
    #print(str(j)+" bp are not AGCTN.")
    return onehot

def ATGCtoArray(char *Seq, int seqlen):
    cdef list onehot=[]
    cdef char Nuc
    cdef int i
    for i in range(seqlen):
        Nuc=Seq[i]
        #print Nuc
        if Nuc=='A' or Nuc=='a':
            onehot.append((1, 0, 0, 0))
        elif Nuc=='T' or Nuc=='t':
            onehot.append((0, 1, 0, 0))
        elif Nuc=='G' or Nuc=='g':
            onehot.append((0, 0, 1, 0))
        elif Nuc=='C' or Nuc=='c':
            onehot.append((0, 0, 0, 1))
        elif Nuc=='N' or Nuc=='n':
            onehot.append((0, 0, 0, 0))
        else:
            print("sequence contains unreadable characters: "+str(Nuc))
            sys.exit()
    
    return onehot

def ACGTtoArray(char *Seq, int seqlen):
    cdef list onehot=[]
    cdef char Nuc
    cdef int i
    for i in range(seqlen):
        Nuc=Seq[i]
        #print Nuc
        if Nuc=='A' or Nuc=='a':
            onehot.append((1, 0, 0, 0))
        elif Nuc=='C' or Nuc=='c':
            onehot.append((0, 1, 0, 0))
        elif Nuc=='G' or Nuc=='g':
            onehot.append((0, 0, 1, 0))
        elif Nuc=='T' or Nuc=='t':
            onehot.append((0, 0, 0, 1))
        elif Nuc=='N' or Nuc=='n':
            onehot.append((0, 0, 0, 0))
        else:
            print("sequence contains unreadable characters: "+str(Nuc))
            sys.exit()
    
    return onehot

def ACGTtoaltArray(char *Seq, int seqlen):
    cdef list onehot=[]
    cdef char Nuc
    cdef int i
    for i in range(seqlen):
        Nuc=Seq[i]
        #print Nuc
        if Nuc=='A' or Nuc=='a':
            onehot.append((1, 1, 0, 0))
        elif Nuc=='C' or Nuc=='c':
            onehot.append((0, 1, 0, 1))
        elif Nuc=='G' or Nuc=='g':
            onehot.append((1, 0, 1, 0))
        elif Nuc=='T' or Nuc=='t':
            onehot.append((0, 0, 1, 1))
        elif Nuc=='N' or Nuc=='n':
            onehot.append((0, 0, 0, 0))
        else:
            print("sequence contains unreadable characters: "+str(Nuc))
            sys.exit()
    
    return onehot

def ACGTto3dArray(char *Seq, int seqlen):
    cdef list onehot=[]
    cdef char Nuc
    cdef int i
    for i in range(seqlen):
        Nuc=Seq[i]
        #print Nuc
        if Nuc=='A' or Nuc=='a':
            onehot.append((0, 0, 1))
        elif Nuc=='C' or Nuc=='c':
            onehot.append((0, 1, 0))
        elif Nuc=='G' or Nuc=='g':
            onehot.append((1, 0, 1))
        elif Nuc=='T' or Nuc=='t':
            onehot.append((1, 1, 0))
        elif Nuc=='N' or Nuc=='n':
            onehot.append((0, 0, 0))
        else:
            print("sequence contains unreadable characters: "+str(Nuc))
            sys.exit()
    
    return onehot