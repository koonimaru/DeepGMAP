import random
import gzip
import cPickle
import numpy as np
import time

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
        
    
    return ''.join(random.choice(str(dnachoice)) for _ in xrange(length))

def AGCTtoArray(Nuc):
    onehot=[]
    if Nuc=="A" or Nuc=="a":
        onehot=np.array([1, 0, 0, 0])
        return onehot
    elif Nuc=="G" or Nuc=="g":
        onehot=np.array([0, 1, 0, 0])
        return onehot
    elif Nuc=="C" or Nuc=="c":
        onehot=np.array([0, 0, 1, 0])
        return onehot
    elif Nuc=="T" or Nuc=="t":
        onehot=np.array([0, 0, 0, 1])
        return onehot
    elif Nuc=="N" or Nuc=="n":
        onehot=np.array([0, 0, 0, 0])
        return onehot
    else: 
        pass

with gzip.open('/media/koh/HD-PCFU3/mouse/filter1_999_Mon Oct 17 19:06:34 2016.cpickle.gz', 'r') as f1:
    learned_variables=cPickle.load(f1)
    filter1=learned_variables[0]
    DNA_seed_array=np.zeros((19, 4, 1, 50000), np.int32)    
    i=0
    j=0
    k=0
    DNA_seed=list()
    for i in range(50000):
        DNA_seed.append(DNA(19, 0.238,0.262,0.262,0.238, 0))
        for j in range(19):
            AGCT_binary=AGCTtoArray(DNA_seed[i][j])
            for k in range(4):
                DNA_seed_array[j][k][0][i]+=AGCT_binary[k]
    i=0
    j=0
    k=0
    l=0
    threshold=0.0
    
    for i in range(512):
        output_file=open('/media/koh/HD-PCFU3/mouse/MotifGenerator_'+str(i)+'.txt', 'w') 
        for j in range(50000):
            threshold=0
            
            for k in range(19):
                for l in range(4):
                    threshold+=filter1[k][l][0][i]*DNA_seed_array[k][l][0][j]
            if threshold>0.60:
                
                output_file.write('>'+str(threshold)+'\n'+str(DNA_seed[j])+'\n')
           
                
        output_file.close()
    
            
        
