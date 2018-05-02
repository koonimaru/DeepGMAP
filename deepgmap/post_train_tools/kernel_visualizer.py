import sys
#import cairocffi as cairo
import cairo
import gzip
import cPickle       
import numpy as np
import os
import math
def _select_color(cr, DNA):
    if DNA=="A":
        cr.set_source_rgb(0, 1, 0)
    elif DNA=="G":
        cr.set_source_rgb(0.8, 0.8, 0)
    elif DNA=="C":
        cr.set_source_rgb(0, 0, 1)
    elif DNA=="T":
        cr.set_source_rgb(1, 0, 0)    
    else:
        cr.set_source_rgb(0.8, 0.8, 0.8)
def seuquence_visualizer(npz_file):

    with np.load(npz_file) as f:
        kernels=f["prediction/W_conv1:0"]
    
    kernel_shape=kernels.shape
    i=0 
    j=0
    k=0
    l=0
    #line_num=10
    #DNA_len=1000
    #max_value=reconstruct.max()
    
    kernels=np.reshape(kernels, (kernel_shape[0], kernel_shape[1],kernel_shape[3]))
    kernel_shape=kernels.shape
    # (9, 4, 320)
    #kernels=np.exp(kernels)
    #kernels=kernels/np.amax(kernels)
    #reconstruct=80*reconstruct/max_value
    #print norm
    #scale_factor=400/max_value
    width=kernel_shape[0]*40+10
    hight=150
    y_center=hight*0.9
    
    
    
    prefix=os.path.splitext(npz_file)[0]+"_kernels/"
    if not os.path.isdir(prefix):
        try:
            os.mkdir(prefix)
        except:
            sys.exit()
    
    meme_fileout=open(prefix+'motifs.meme','w')
    
        
    for k in range(kernel_shape[2]):
        meme_def="MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n\
Background letter frequencies (from uniform background):\nA 0.2500 C 0.2500 G 0.2500 T 0.2500\n\nMOTIF kernel_"+str(k)+"\n\nletter-probability matrix: alength= 4 w= 1000 nsites= 20 E= 0\n"
        meme_fileout.write(meme_def)
        ims1 = cairo.PDFSurface(None, width, hight)       
        cr = cairo.Context(ims1)
        cr.set_source_rgb(0.0,0.0,0)
        cr.move_to(width*0.1, y_center)
        cr.line_to(width*0.9, y_center)
        #cr.move_to(50, 100)
        #cr.line_to(50, 412)
        cr.set_line_width(2)
        cr.stroke()
        cr.move_to(width*0.1, y_center)
        cr.line_to(width*0.1, y_center-120)
        cr.set_line_width(2)
        cr.stroke()
        cr.move_to(width*0.1, y_center-60)
        cr.line_to(width*0.08, y_center-60)
        cr.set_line_width(2)
        cr.stroke()
        cr.move_to(width*0.075, y_center-60+4*10)
        cr.rotate(-90*math.pi/180.0)
        cr.set_line_width(2)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
        font_mat=cairo.Matrix(xx=32.0,yx=0.0,xy=0.0,yy=32,x0=0.0,y0=0.0)
        cr.set_font_matrix(font_mat)
        cr.show_text("2 bit")
        cr.rotate(90*math.pi/180.0)
        #print y_center
            
        AGCT={}
        #values=[]
        #print reconstruct[k]
        
        #reconstruct[k]=reconstruct[k]*50
        #mean=np.mean(reconstruct[k])
        #stdv=np.std(reconstruct[k])
        
        #reconstruct[k]=(reconstruct[k]-mean)/stdv
        #print kernels[:,:,k]
        xkernel=kernels[:,:,k]
        xkernel=np.exp(xkernel*50.0)
        #print xkernel
        #sys.exit()
        probability=xkernel/np.nansum(xkernel, axis=1)[:,None]
        #probability/=np.nansum(probability)
        for p in probability:        
            to_print=str(p[0])+" "+str(p[2])+" "+str(p[1])+" "+str(p[3])+"\n"
            meme_fileout.write(to_print)
        meme_fileout.write("\n")
        
        for pind, p in enumerate(probability):
            
            ic=np.nansum(p*np.log2(p*4+0.0001))*80
            #print ic
            A=["A", p[0]*ic]
            G=["G",p[1]*ic]
            C=["C",p[2]*ic]
            T=["T", p[3]*ic]
            values=[A,G,C,T]
            pos=filter(lambda x:x[1]>=0,values)
            #neg=filter(lambda x:x[1]<0,values)
            pos.sort(key=lambda x:x[1])
            #neg.sort(key=lambda x:x[1], reverse=True)
            Nucpos=0.01
            #Nucneg=0
            x_pos=width*0.1+pind*30
        
            for l in range(len(pos)):
                Nuc=pos[l][0]
                
                Nucsize=pos[l][1]+0.01
                cr.move_to(x_pos, y_center-Nucpos*0.75)               
                cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
                _select_color(cr, Nuc)
                font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=Nucsize,x0=0.0,y0=0.0)
                cr.set_font_matrix(font_mat)
                cr.show_text(str(Nuc))
                Nucpos+=abs(pos[l][1])
        ims1.write_to_png(prefix+"kernel_"+str(k)+'.png')
        cr.show_page()
    meme_fileout.close()
    

def main():
    npz_file='/home/fast/onimaru/data/output/deepshark_trained_variables_Sat_Apr_28_170548_2018.npz'
    a=npz_file.split('/')[-1]
    a=a.split('.')[0]
    #output_file='/home/fast/onimaru/data/output/deepshark_trained_variables_Sat_Apr_28_170548_2018.npz'
    seuquence_visualizer(npz_file)
if __name__ == "__main__":    
    main()
    