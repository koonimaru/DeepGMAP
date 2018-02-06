
cimport cython
import sys
def label_reader(file_):
    cdef list label_position=[]
    cdef list label_list=[]
    cdef list line_=[]
    cdef int i=0
    cdef str line
    for line in file_:
        if "#" in line:
            continue
        if not line[0:5]=='chr2\t':
            line_=line.split()
            label_position.append(str(line_[0])+':'+str(line_[1])+'-'+str(line_[2]))
            label_list.append(tuple(map(int, line_[3:])))
        else:
            sys.stdout.write('skipping '+line[0:5]+'\r')
            sys.stdout.flush()
        i+=1
        if i%100000==0:
            sys.stdout.write('reading labels %i \r' % (i))
            sys.stdout.flush()
    return label_position, label_list

def AGCTtoArray3(char *Seq, int seqlen):
    cdef list onehot=[]
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

