cimport cython
import sys
import numpy as np
from scipy.spatial.distance import cdist

def narrowPeak_writer(str out_dir,list y_prediction2,list position_list):
    cdef str filename_1
    filename_1=out_dir+'.narrowPeak'
    print('writing '+filename_1)
    output_handle=open(filename_1, 'w')
    cdef int k=0
    cdef float value
    cdef str chrom, start_, end_
    for i in range(len(y_prediction2)):

        a=position_list[i].strip('>')
        #print(str(a)+'\t'+str(y_prediction2[i]))
        k+=1
        a=a.split(':')
        chrom=a[0]
        b=a[1].split('-')
        start_=b[0]
        end_=b[1]
        value=y_prediction2[i]
        output_handle.write(str(chrom)+'\t'
                            +str(start_)+'\t'
                            +str(end_)+'\t.\t'
                            +str(value*1000).strip('[]')+'\t.\t'
                            +str(value).strip('[]')+"\t-1\t-1\t-1\n")
            
    print("prediction num: "+str(k))
    output_handle.close()
    print('finished writing '+filename_1)
    
    
def motif_compare(motif_data_dict, long_motif_dict, fout, THRESHOLD=-5.0):
    cdef int i,k, j, l=0
    cdef str k1, k2
    cdef double RAND_MEAN, RAND_DEV,DIST,Z_SCORE, ic
    cdef list comp_result, comp_result2, cpr, Z_SCORE_list=[]
    cdef int[2] v2shape,v1shape
    #cdef double[4] pm1
    with open(fout, "w") as f:
        comp_result2=[]
        f.write("Motif name\tStart\tEnd\tDistance\n")
        for k1, v1 in long_motif_dict.items():
            
            v1shape=v1.shape
            #print v1shape
            for k2, v2 in motif_data_dict.items():
                ic1=0
                if "secondary" in k2:
                    continue
                #print k2
                #j+=1
                #print j
                v2shape=v2.shape
                #print v2shape
                """for i in range(v2shape[0]):
                    ic=np.nansum(v2[i]*np.log2(v2[i]*4+0.000001))
                    v2[i]=v2[i]"""
 
                RAND_DIST=np.zeros([500], np.float32)
                for i in range(500):
                    rand=np.random.rand(v2shape[0],v2shape[1])
                    for k in range(v2shape[0]):
                        rand[k]=rand[k]/np.sum(rand[k])
                        #rand[k]=pm1*(np.sum(pm1*np.log2(pm1*4+0.00001)))
                    #RAND_DIST.append(np.mean(np.diagonal(cdist(v2, rand,metric='euclidean'))))
                    M=0.5*(rand+v2)+0.00001
                    DIST=0.5*(np.sum(-v2*np.log(M/(v2+0.00001)))+np.sum(-rand*np.log(M/(rand+0.00001))))/float(v2shape[0])
                    #DIST=-np.sum(v2*np.log(rand+0.00001)+(1.0-v2)*np.log(1.0-rand+0.00001))/float(v2shape[0])
                    RAND_DIST[i]+=DIST
                    
                RAND_MEAN=np.mean(RAND_DIST)
                RAND_DEV=np.std(RAND_DIST)
                #print RAND_MEAN, RAND_DEV
                #print("randome_dist: "+str(RAND_DIST))
                comp_result=[]
                for i in range(v1shape[0]-v2shape[0]):
                    #partial_motif=[]
                    #for j in range(v2shape[0]):
                    #    pm1=v1[i+j]
                        #ic=np.sum(pm1*np.log2(pm1*4+0.000001))
                    partial_motif_=v1[i:i+v2shape[0]]
                    #partial_motif_=np.array(partial_motif)
                    #partial_motif=v1[i:(i+v2shape[0])]
                    #print v2shape, np.shape(partial_motif)
                    M=0.5*(partial_motif_+v2)+0.00001
                    DIST=0.5*(np.sum(-v2*np.log(M/(v2+0.00001)))+np.sum(-partial_motif_*np.log(M/(partial_motif_+0.00001))))/float(v2shape[0])
                    #print JSD
                    v2_comp=np.flip(np.flip(v2,0),1)
                    M_comp=0.5*(partial_motif_+v2_comp)+0.00001
                    DIST_comp=0.5*(np.sum(-v2_comp*np.log(M_comp/(v2_comp+0.00001)))+np.sum(-partial_motif_*np.log(M_comp/(partial_motif_+0.00001))))/float(v2shape[0])
                    #DIST=np.mean(np.diagonal(cdist(v2, partial_motif_,metric='euclidean')))
                    #DIST=np.mean(np.diagonal(cdist(v2, partial_motif_,metric='euclidean')))
                    #DIST_comp=np.mean(np.diagonal(cdist(v2_comp, partial_motif_,metric='euclidean')))
                    
                    #DIST=-np.sum(v2*np.log(partial_motif_+0.00001)+(1.0-v2)*np.log(1.0-partial_motif_+0.00001))/float(v2shape[0])
                    #DIST_comp=-np.sum(v2_comp*np.log(partial_motif_+0.00001)+(1.0-v2_comp)*np.log(1.0-partial_motif_+0.00001))/float(v2shape[0])
                    ori="+"
                    if DIST_comp<DIST:
                        DIST=DIST_comp
                        ori="-"
                    Z_SCORE=(DIST-RAND_MEAN)/RAND_DEV
                    print Z_SCORE
                    Z_SCORE_list.append(Z_SCORE)
                    if Z_SCORE<=THRESHOLD:
                        l+=1
                        #f.write(str(k2)+"\t"+str(i)+"\t"+str(i+v2shape[0])+"\t"+str(Z_SCORE)+"\n")
                        comp_result.append([k2,i,i+v2shape[0],Z_SCORE,ori])
                #print comp_result
                    
                if 0<len(comp_result)<10:
                    for cpr in comp_result:
                        comp_result2.append(cpr)
                        #f.write(str(cpr[0])+"\t"+str(cpr[1])+"\t"+str(cpr[2])+"\t"+str(cpr[3])+"\n")
                elif len(comp_result)>=10:
                    comp_result.sort(key = lambda x: x[3])
                    for cpr in comp_result[-10:]:
                        comp_result2.append(cpr)
                        
        comp_result2.sort(key = lambda x: x[1])
        for cpr in comp_result2:
            f.write("\t".join([str(cpr[0]),str(cpr[1]),str(cpr[2]),str(cpr[3]),str(cpr[4])])+"\n")
                    
                        
        print("the number of motif matches: "+str(l))
    return Z_SCORE_list
                        
                        
                        
                        
                        
                        
