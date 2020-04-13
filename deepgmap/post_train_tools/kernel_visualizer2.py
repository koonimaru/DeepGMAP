import sys
import cairocffi as cairo
#import cairo
#from gi.repository import Gtk

import gzip
import cPickle       
import numpy as np
import os
import math
import enum
from PIL import Image
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
def _select_color(cr, DNA):
    if DNA=="A":
        cr.set_source_rgb(1, 0, 0)
    elif DNA=="G":
        cr.set_source_rgb(0.8, 0.8, 0)
    elif DNA=="C":
        cr.set_source_rgb(0, 0, 1)
    elif DNA=="T":
        cr.set_source_rgb(0, 1, 0)    
    else:
        cr.set_source_rgb(0.8, 0.8, 0.8)
        
def rectangle(x, y, w,h,lw, context):
    context.set_line_width(lw)
    context.move_to(x, y)
    context.rel_line_to(w, 0)
    context.rel_line_to(0, h)
    context.rel_line_to(-w, 0)
    context.close_path()

def seuquence_visualizer(npz_file):
    png_list=[]
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
    y_center=hight*0.8
    
    
    
    prefix=os.path.splitext(npz_file)[0]+"_kernels/"
    if not os.path.isdir(prefix):
        try:
            os.mkdir(prefix)
        except:
            sys.exit()
    
    meme_fileout=open(prefix+'motifs.meme','w')
    meme_def="MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n\Background letter frequencies (from uniform background):\nA 0.2500 C 0.2500 G 0.2500 T 0.2500\n\n"
    meme_fileout.write(meme_def)
    kernel_shape_ic_list=[]
    for k in range(kernel_shape[2]):
        meme_def="MOTIF kernel_"+str(k)+"\n\nletter-probability matrix: alength= 4 w= 9 nsites= 9 E= 0\n"
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
        #cr.set_line_width(2)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
        font_mat=cairo.Matrix(xx=32.0,yx=0.0,xy=0.0,yy=32,x0=0.0,y0=0.0)
        cr.set_font_matrix(font_mat)
        cr.show_text("2 bit")
        cr.rotate(90*math.pi/180.0)
        font_mat=cairo.Matrix(xx=12.0,yx=0.0,xy=0.0,yy=12,x0=0.0,y0=0.0)
        cr.move_to(width*0.5, hight)
        cr.show_text("k"+str(k))
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
        xkernel=np.exp(xkernel*100.0)
        #print xkernel
        #sys.exit()
        probability=xkernel/np.nansum(xkernel, axis=1)[:,None]
        #probability/=np.nansum(probability)
        for p in probability:        
            to_print=str(p[0])+" "+str(p[2])+" "+str(p[1])+" "+str(p[3])+"\n"
            meme_fileout.write(to_print)
        meme_fileout.write("\n\n")
        ic_sum=0.0
        for pind, p in enumerate(probability):
            
            ic=np.nansum(p*np.log2(p*4+0.0001))*80
            ic_sum+=ic
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
        png_list.append(prefix+"kernel_"+str(k)+'.png')
        kernel_shape_ic_list.append(ic_sum)
        cr.show_page()
    meme_fileout.close()
    
    return png_list, kernel_shape_ic_list

def kernel_connector(png_list):
    
    
    
    
    pt_per_mm = 72 / 25.4
    width, height = 210 * pt_per_mm, 297 * pt_per_mm
    upper_lim=height*0.1
    lateral_lim=width*0.1
    x_interval=width*0.8/9.0
    nodew=width*0.005 
    nodeh=0.0015*height
    
    out_dir=os.path.split(png_list[0])[0]+"/kernels.pdf"
    ims1 = cairo.PDFSurface(out_dir, width, height)
    cr = cairo.Context(ims1)
    #cr.scale(pt_per_mm,pt_per_mm)
    coordinates={}
    im = Image.open(png_list[0])
    xwidth=int(width*0.8/10.0)+5
    ywidth=int(im.size[1]*xwidth/float(im.size[0]))
    #print xwidth, ywidth
    for k, i in enumerate(png_list):
        im = Image.open(i)
        
        im=im.resize([xwidth, ywidth], Image.ANTIALIAS)
        #im.thumbnail([xwidth, ywidth], Image.LANCZOS)
        _buffer = StringIO.StringIO()
        im.save(_buffer, format="PNG", quality=100)#,compress_level=0, dpi=(xwidth, ywidth))
        _buffer.seek(0)
        png_image=cairo.ImageSurface.create_from_png(_buffer)
        cr.save()
        #cr.scale(0.5, 0.5)
        cr.set_source_surface(png_image, lateral_lim+(xwidth-5)*(k%10), upper_lim+ywidth*((k/10)))
        cr.paint()

    cr.show_page()
        

def main():
    if len(sys.argv)>1:
        npz_file=sys.argv[1]
    else:
        #npz_file='/home/fast/onimaru/deepgmap/data/outputs/conv4frss_trained_variables_Fri_May_11_075425_2018.npz'
        npz_file='/home/fast2/onimaru/DeepGMAP-dev/data/outputs/conv4frss_Mon_Feb_25_092345_2019_trained_variables.npz'
    #output_file='/home/fast/onimaru/data/output/deepshark_trained_variables_Sat_Apr_28_170548_2018.npz'
    
    png_list, kernel_shape_ic_list=seuquence_visualizer(npz_file)
    kernel_connector(png_list)
if __name__ == "__main__":    
    main()
    