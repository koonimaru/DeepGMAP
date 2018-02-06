import random
import gzip
import cPickle
import numpy as np
import time


with gzip.open('/media/koh/HD-PCFU3/mouse/filter1_1499.cpickle.gz', 'r') as f1:
    learned_variables=cPickle.load(f1)
    filter1=learned_variables[0]
    bottom_up_filter1=np.zeros((384, 19, 4), np.int32)    
    i=0
    j=0
    k=0
    l=0
    threshold=0.0
    output_handle=open('/media/koh/HD-PCFU3/mouse/filter1_1499.cpickle.gz', 'w')
    for i in range(384):
        output_handle.write('#filter'+str(i)+'\n')           
        for k in range(19):
            a=np.zeros((4), np.float32)
            for l in range(4):
                a[l]=filter1[k][l][0][i]
            minimum_value=np.amin(a)
            if minimum_value<0:
                a+=-minimum_value
            sum_value=np.sum(a)
            norm_a=10*(a/sum_value)
            DNA_seq=''
            m=0
            for m in range(int(norm_a[0])):
                DNA_seq+='A'
            m=0
            for m in range(int(norm_a[1])):
                DNA_seq+='G'
            m=0
            for m in range(int(norm_a[1])):
                DNA_seq+='C'
            m=0
            for m in range(int(norm_a[1])):
                DNA_seq+='T'    
            output_handle.write(str(DNA_seq)+'\n')
            
           
            
    
           
                

            
        
