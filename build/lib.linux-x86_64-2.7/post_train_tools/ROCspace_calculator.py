import numpy as np

    

def Three_label_ROCspace_calculator(a, b):
    True_positive1=[2,0, 0]
    True_positive2=[0,2, 0]
    True_negative=[0, 0, 2]
    
    False_postive1=[-1, 0, 1]
    False_postive2=[0, -1, 1]
    False_negative1=[1, 0, -1]
    False_negative2=[0, 1, -1]
            
    Wrong_assignment1=[-1, 1, 0]
    Wrong_assignment2=[1, -1, 0]
    
    ROC_counter=np.zeros((9),np.int32) #index1 = True positive, index2=False positive, index3=False negative, index3= True negative
    for i in range(len(a)):
        b1=[0,0,0]
        index=np.argmax(b[i])
        b1[index]+=1
               
        c1=a[i]+b1
        c2=a[i]-b1
        
        if (c1==True_positive1).all():
            ROC_counter[0]+=1
        elif (c1==True_positive2).all():
            ROC_counter[1]+=1
        elif (c1==True_negative).all():
            ROC_counter[2]+=1
        elif (c2==False_postive1).all():
            ROC_counter[3]+=1
        elif (c2==False_postive2).all():
            ROC_counter[4]+=1
        elif (c2==False_negative1).all():
            ROC_counter[5]+=1
        elif (c2==False_negative2).all():
            ROC_counter[6]+=1
        elif (c2==Wrong_assignment1).all():
            ROC_counter[7]+=1
        elif (c2==Wrong_assignment2).all():
            ROC_counter[8]+=1  
    FPR=float(ROC_counter[3]+ROC_counter[4])/(float(ROC_counter[2]+ROC_counter[3]+ROC_counter[4]))
    FNR=float(ROC_counter[5]+ROC_counter[6])/(float(ROC_counter[0]+ROC_counter[1]+ROC_counter[5]+ROC_counter[6]))
    TSS_detection_rate=float(ROC_counter[1])/(float(ROC_counter[6]+ROC_counter[1]+ROC_counter[7]))
    Wrong_assignment=float(ROC_counter[7]+ROC_counter[8])/(float(ROC_counter[3]+ROC_counter[4]+ROC_counter[5]+ROC_counter[6]+ROC_counter[7]+ROC_counter[8])+0.000001) 
    return FPR, FNR, TSS_detection_rate, Wrong_assignment

def Four_label_ROCspace_calculator(a, b):
    True_positive1=[2,0, 0, 0]
    True_positive2=[0,2, 0, 0]
    True_positive3=[0,0, 2, 0]
    True_negative=[0, 0, 0, 2]
    
    
    False_postive1=[-1, 0,0, 1]
    False_postive2=[0, -1,0, 1]
    False_postive3=[0, 0,-1, 1]
    False_negative1=[1, 0, 0,-1]
    False_negative2=[0, 1,0, -1]
    False_negative3=[0, 0,1, -1]        
    Wrong_assignment11=[-1, 1, 0,0]
    Wrong_assignment12=[0, 1, -1,0]
    Wrong_assignment21=[1, -1, 0,0]
    Wrong_assignment22=[1, 0, -1,0]
    Wrong_assignment31=[-1, 0, 1,0]
    Wrong_assignment32=[0, -1, 1,0]
    
    ROC_counter=np.zeros((9),np.int32)
     #index1 = True positive, index2=False positive, index3=False negative, index3= True negative
    for i in range(len(a)):
        b1=[0,0,0,0]
        index=np.argmax(b[i])
        b1[index]+=1
               
        c1=a[i]+b1
        c2=a[i]-b1
        
        if (c1==True_positive1).all():
            ROC_counter[0]+=1
        elif (c1==True_positive2).all():
            ROC_counter[1]+=1
        elif (c1==True_negative).all():
            ROC_counter[2]+=1
        elif (c2==False_postive1).all():
            ROC_counter[3]+=1
        elif (c2==False_postive2).all():
            ROC_counter[4]+=1
        elif (c2==False_negative1).all():
            ROC_counter[5]+=1
        elif (c2==False_negative2).all():
            ROC_counter[6]+=1
        elif (c2==Wrong_assignment11).all():
            ROC_counter[7]+=1
        elif (c2==Wrong_assignment21).all():
            ROC_counter[8]+=1
          
    FPR=float(ROC_counter[3]+ROC_counter[4])/(float(ROC_counter[2]+ROC_counter[3]+ROC_counter[4]))
    FNR=float(ROC_counter[5]+ROC_counter[6])/(float(ROC_counter[0]+ROC_counter[1]+ROC_counter[5]+ROC_counter[6]))
    TSS_detection_rate=float(ROC_counter[1])/(float(ROC_counter[6]+ROC_counter[1]+ROC_counter[7]))
    Wrong_assignment=float(ROC_counter[7]+ROC_counter[8])/(float(ROC_counter[3]+ROC_counter[4]+ROC_counter[5]+ROC_counter[6]+ROC_counter[7]+ROC_counter[8])+0.000001) 
    return FPR, FNR, TSS_detection_rate, Wrong_assignment