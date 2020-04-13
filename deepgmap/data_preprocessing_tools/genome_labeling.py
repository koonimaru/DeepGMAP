
import glob as glb
import sys
import numpy as np

narrow_peak_list=glb.glob(sys.argv[1])
label_array=[]
for i in range(len(narrow_peak_list)):
    label_array.append(0)
#print label_array

genome_file=sys.argv[2]
genome_segments=[]

with open(genome_file, 'r') as fin:
    for line in fin:
        #label_array.append(1)
        genome_segments.append(label_array)
genome_segments=np.array(genome_segments)

genome_file2=sys.argv[3]
genome_size=[]
chrm_num={}
chrm_list=set()
with open(genome_file2,'r') as fin:
    i=0
    for line in fin:
        line=line.split()
        genome_size.append(int(line[1])/1000)
        chrm_num[line[0]]=i
        chrm_list.add(line[0])
        i+=1
        

peak_dict={}

i=0
j=0
for f in narrow_peak_list:
    with open(f, 'r') as fin:
        for line in fin:
            
            line=line.split()
            chrm=line[0]
            if len(line)==4:                
                score=int(line[3])
            elif len(line)==10:
                score=int(line[4])
            else:
                #print f
                break
            
            if chrm in chrm_list:
                if score>=75:
                    
                    start=int(line[1])
                    end=int(line[2])
                    length=end-start
                    right_bin1=(start/1000+1)*1000
                    left_bin1=(end/1000)*1000
                    point_1000=(start+end)/(2*1000)
                    if end<=right_bin1:
                        
                        
                        genome_location=sum(genome_size[:chrm_num[chrm]])+point_1000
                        if genome_segments[genome_location][i]==0:
                            genome_segments[genome_location][i]+=1
                            j+=1
                            #print j
                    else:
                        
                        
                        if right_bin1-start>=100:
                            left_point=start/1000
                        else:
                            left_point=start/1000+1
                        if end-left_bin1>=100:
                            right_point=end/1000
                        else:
                            right_point=end/1000-1
                        k=left_point
                        while left_point<=k<=right_point:
                            genome_location=sum(genome_size[:chrm_num[chrm]])+k
                        
                            if genome_segments[genome_location][i]==0:
                                genome_segments[genome_location][i]+=1
                                j+=1
                                #print j
                                #print "longer than 1000 "+ str(genome_location)
                            k+=1
                        
                #if genome_segments[genome_location][-1]==1:
                #    genome_segments[genome_location][-1]=0
    i+=1

with open(genome_file,'r') as fin:
    with open(genome_file+'_limb_75co_OL100.labeled','w') as fout, open(genome_file+'_limb_75co_OL100.bed','w') as fout2:
        i=0
        for line in fin:
            fout.write(line.strip('\n')+'\t'+'\t'.join(map(str, list(genome_segments[i])))+'\n')
            if genome_segments[i]==1:
                fout2.write(line)
            i+=1
            
            

        

