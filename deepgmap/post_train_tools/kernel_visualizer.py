import sys
import cairocffi as cairo
#import cairo
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

def kernel_connector(npz_file, png_list, kernel_shape_ic_list):
    
    
    """
    variable_dict={"W_conv1": W_conv1, 
                   "W_conv2": W_conv2,
                   "W_conv21": W_conv21, 
                   "W_conv22": W_conv22, 
                   "W_fc1": W_fc1,
                   "W_fc4": W_fc4, 
                   "b_fc1": b_fc1, 
                   "b_fc4": b_fc4}
    """
    
    conv_kernels=[]
    fc_kernels=[]
    with np.load(npz_file) as f:
        #conv_kernels.append(f["prediction/W_conv1:0"])
        conv_kernels.append(f["prediction/W_conv2:0"])
        conv_kernels.append(f["prediction/W_conv21:0"])
        conv_kernels.append(f["prediction/W_conv22:0"])
        fc_kernels.append(f["prediction/W_fc1:0"])
        fc_kernels.append(f["prediction/W_fc4:0"])
    conv_kernels_sum=[]
    for i in conv_kernels:
        i_shape=i.shape
        #print i_shape
        i_reshape=i.reshape([i_shape[0],i_shape[2],i_shape[3]])
        conv_kernels_sum.append(np.clip(np.sum(i_reshape, axis=0), 0.0, None))
    #print fc_kernels[0].shape, conv_kernels[-1].shape
    last_seq_length=fc_kernels[0].shape[0]/conv_kernels[-1].shape[-1]
    #print last_seq_length
    #sys.exit()
    pt_per_mm = 72 / 25.4
    width, height = 210 * pt_per_mm, 297 * pt_per_mm*2
    upper_lim=height*0.05
    lateral_lim=width*0.4
    x_interval=width*0.8/9.0
    nodew=width*0.005 
    nodeh=0.0015*height
    
    y_num=fc_kernels[-1].shape[1]
    for _y_num in range(y_num):
        out_dir=os.path.splitext(npz_file)[0]+"_"+str(_y_num)+"_network.pdf"
        ims1 = cairo.PDFSurface(out_dir, width, height)
        cr = cairo.Context(ims1)
        coordinates={}
        """drawing nodes in convolution layers on a surface"""
        
        for i, conv in enumerate(conv_kernels_sum):
            x=lateral_lim+i*(x_interval)
            cr.move_to(x, upper_lim-0.0018*height)
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
            font_mat=cairo.Matrix(xx=12.0,yx=0.0,xy=0.0,yy=12,x0=0.0,y0=0.0)
            cr.set_font_matrix(font_mat)
            cr.show_text("hidden"+str(i+1))
            coordinates["hidden"+str(i+1)]=[]
            conv_shape=conv.shape
            print(conv_shape)
            for j in range(conv_shape[0]):
                y=upper_lim+0.0018*height*j
                #cr.move_to(width*0.1, height*0.1+0.0008*height*j)
                
                #cr.arc(x, y, 0.001*height, 0, 2*math.pi)
                cr.rectangle(x, y,nodew, nodeh)
                coordinates["hidden"+str(i+1)].append([x,y])
                cr.fill()
            if i+1==len(conv_kernels_sum):
                x+=x_interval
                cr.move_to(x, upper_lim-0.0018*height)
                cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
                font_mat=cairo.Matrix(xx=12.0,yx=0.0,xy=0.0,yy=12,x0=0.0,y0=0.0)
                cr.set_font_matrix(font_mat)
                cr.show_text("hidden"+str(i+2))
                coordinates["hidden"+str(i+2)]=[]
                last_conv_y=0.0
                for j in range(conv_shape[1]):
                    
                    y=upper_lim+0.0018*height*j
                    #cr.move_to(width*0.1, height*0.1+0.0008*height*j)
                    
                    #cr.arc(x, y, 0.001*height, 0, 2*math.pi)
                    cr.rectangle(x, y,nodew, nodeh)
                    coordinates["hidden"+str(i+2)].append([x,y])
                    cr.fill()
                
                last_conv_y=y+0.0018*height-upper_lim
                last_conv_shape=conv_shape[1]
        
             
        """drawing nodes as bars in fully-connected layers"""
        
        for i, fc in enumerate(fc_kernels):
            x+=x_interval
            cr.move_to(x, upper_lim-0.0018*height)
            cr.set_source_rgba(0.0,0.0,0.0,1.0)
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
            font_mat=cairo.Matrix(xx=12.0,yx=0.0,xy=0.0,yy=12,x0=0.0,y0=0.0)
            cr.set_font_matrix(font_mat)
            cr.show_text("fc"+str(i+1))
            coordinates["fc"+str(i+1)]=[]
            fc_shape=fc.shape
            print(fc_shape)
            y=upper_lim
            #cr.move_to(width*0.1, height*0.1+0.0008*height*j)
            
            #cr.arc(x, y, 0.001*height, 0, 2*math.pi)
            cr.set_source_rgba(0.5,0.5,0.5,0.5)
            cr.rectangle(x, y,width*0.015, fc_shape[0]*last_conv_y/(float(last_seq_length)*last_conv_shape))
            coordinates["fc"+str(i+1)].append([x,y])
            cr.fill()
    
        fc2_hight=fc_shape[0]*last_conv_y/(float(last_seq_length)*last_conv_shape)
        """drawing nodes in the last prediction layer"""
        x+=x_interval
        cr.move_to(x, upper_lim-0.0018*height)
        cr.set_source_rgba(0.0,0.0,0.0,1.0)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
        font_mat=cairo.Matrix(xx=12.0,yx=0.0,xy=0.0,yy=12,x0=0.0,y0=0.0)
        cr.set_font_matrix(font_mat)
        cr.show_text("y")
        coordinates["y"]=[]
        fc_shape=fc.shape
    
        
        for j in range(fc_shape[1]):
                
                y=upper_lim+fc2_hight/float(fc_shape[1])*j
                cr.set_source_rgba(0.0,0.0,0.0,1.0)
                cr.rectangle(x, y,nodew*3, nodeh*3)
                coordinates["y"].append([x,y])
                cr.fill()
    
        
        w_fc2=fc_kernels[1]
        w_fc2_max=np.amax(w_fc2)
        w_fc2_shape=w_fc2.shape
        
        x2, y2=coordinates["fc2"][0]
        new_fc2_coord=[]
        for k in range(w_fc2_shape[0]):
            y2_=y2+(last_conv_y/(float(last_seq_length)*last_conv_shape))*k
            new_fc2_coord.append([x2, y2_])
        
        nodes_interest=_y_num
        
        linked=set()
        unfilterd_links=[]
        for i, j in enumerate(w_fc2) :
            #x1, y1=new_fc2_coord[i]
            for k in range(len(j)):
                if k==nodes_interest:
                    x2, y2=coordinates["y"][k]
                    cr.set_source_rgba(1,0,0,1)
                    cr.rectangle(x2, y2,nodew*3, nodeh*3)
                    cr.fill()
                    unfilterd_links.append([j[k], i, k])
        rev_sort=sorted(unfilterd_links, reverse=True)
        if len(rev_sort)<10:
            _limit=len(rev_sort)
        else:
            _limit=10
        for m, i, k in rev_sort[:_limit]:
            x1, y1=new_fc2_coord[i]
            x2, y2=coordinates["y"][k]
            cr.move_to(x1+width*0.005, y1+0.001*height/2.0)
            cr.line_to(x2, y2+0.001*height/2.0)
            cr.set_source_rgba(0.0, 0.0,1.0,0.5)
            cr.set_line_width(1)
            cr.stroke()
            linked.add(i)
                    
        
        w_fc1=fc_kernels[0]
        w_fc1_max=np.amax(w_fc1)
        w_fc1_shape=w_fc1.shape
        print(w_fc1_shape)
        
        linked2=set()
        unfilterd_links=[]
        for i, j in enumerate(w_fc1) :
            
            for k in range(len(j)):
                if k in linked:
                    unfilterd_links.append([j[k],i/last_seq_length, k])
        rev_sort=sorted(unfilterd_links, reverse=True)
        print(rev_sort[:5])
        if len(rev_sort)<conv_kernels[-1].shape[-1]/10:
            _limit=len(rev_sort)
        else:
            _limit=conv_kernels[-1].shape[-1]/10
        for m, i, k in rev_sort[:_limit]:
            x1, y1=coordinates["hidden4"][i]
            
            cr.move_to(x1+width*0.005+x_interval, y1+0.001*height/2.0)
            cr.line_to(*new_fc2_coord[k])
            cr.set_source_rgba(0.0, 0.0,1.0,0.5)
            cr.set_line_width(1)
            cr.stroke()
            cr.move_to(x1+width*0.005+x_interval, y1+0.001*height/2.0)
            cr.line_to(x1+width*0.005, y1+0.001*height/2.0)
            cr.set_source_rgba(0.5, 0.5,0.5,0.5)
            cr.set_line_width(1)
            cr.stroke()
            linked2.add(i)
        
        linked_conv=[]
        linked_conv.append(linked2)
        for i, conv in reversed(list(enumerate(conv_kernels_sum))):
            
            conv_shape=conv.shape
            print(i, conv_shape)
            convf=conv.flatten()
            convf=convf[(-convf).argsort()]
           # topN=convf[600]
            max_value=np.amax(conv)
            linked3=set()
            unfilterd_links=[]
            for j in range(conv_shape[1]):
                
                #if i==0:
                    #k=kernel_shape_ic_list
                
                k=conv[:,j]
                #idx = (-k).argsort()[:10]
                #print k.shape
                #max_value=np.amax(k)
                
                for l, m in enumerate(k):
                    if j in linked_conv[-1]:
                        unfilterd_links.append([m, j, l])
            rev_sort=sorted(unfilterd_links, reverse=True)
            if len(rev_sort)<conv_shape[0]/10:
                _limit=len(rev_sort)
            else:
                _limit=conv_shape[0]/10
            for m, j, l in rev_sort[:_limit]:
                x1, y1=coordinates["hidden"+str(i+2)][j]
                cr.set_source_rgba(1,0,0,1)
                cr.rectangle(x1, y1,nodew, nodeh)
                cr.fill()
                #if coordinates.has_key("hidden"+str(i)):
                x2, y2=coordinates["hidden"+str(i+1)][l]
                cr.set_source_rgba(1,0,0,1)
                cr.rectangle(x2, y2,nodew, nodeh)
                cr.fill()
                cr.move_to(x1, y1+0.001*height/2.0)
                cr.line_to(x2+width*0.005, y2+0.001*height/2.0)
                cr.set_source_rgba(0, 0,1.0,0.5)
                cr.set_line_width(1)
                cr.stroke()
                linked3.add(l)
            if len(linked3)==0:
                sys.exit("no link was found")
            linked_conv.append(linked3)
        prev_y1=[]
        for i in sorted(linked_conv[-1]):
            x1, y1=coordinates["hidden1"][i]
            im = Image.open(png_list[i])
            xwidth=128
            ywidth=int(im.size[1]*xwidth/float(im.size[0]))
            im=im.resize([xwidth, ywidth], Image.ANTIALIAS)
            _buffer = StringIO.StringIO()
            im.save(_buffer, format="PNG",quality=100)
            _buffer.seek(0)
            png_image=cairo.ImageSurface.create_from_png(_buffer)
            if not len(prev_y1)==0 and (y1-prev_y1[-1][1])<(ywidth-10) and prev_y1[-1][0]==0:
                a=xwidth-20
            else:
                a=0
            cr.set_source_surface(png_image, lateral_lim-xwidth-a+10, y1-ywidth/2)
            cr.paint()
            prev_y1.append([a, y1])
        
        cr.show_page()
        

def main():
    if len(sys.argv)>1:
        npz_file=sys.argv[1]
    else:
        #npz_file='/home/fast/onimaru/deepgmap/data/outputs/conv4frss_trained_variables_Fri_May_11_075425_2018.npz'
        npz_file='/home/fast2/onimaru/DeepGMAP-dev/data/outputs/conv4frss_Mon_Feb_25_092345_2019_trained_variables.npz'
        #npz_file='/home/fast/onimaru/deepgmap/data/outputs/conv4frss_Wed_May_30_174635_2018_trained_variables_16000_.npz'
    #output_file='/home/fast/onimaru/data/output/deepshark_trained_variables_Sat_Apr_28_170548_2018.npz'
    png_list, kernel_shape_ic_list=seuquence_visualizer(npz_file)
    kernel_connector(npz_file, png_list, kernel_shape_ic_list)
if __name__ == "__main__":    
    main()
    