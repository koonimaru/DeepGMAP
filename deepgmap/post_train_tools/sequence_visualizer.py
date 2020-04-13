
import cairocffi as cairo
import gzip
import cPickle       
import numpy as np
import glob as gl

def select_color(cr, DNA):
    if DNA=="A":
        cr.set_source_rgb(1, 0, 0)
    elif DNA=="G":
        cr.set_source_rgb(0, 0, 0)
    elif DNA=="C":
        cr.set_source_rgb(0, 0, 1)
    elif DNA=="T":
        cr.set_source_rgb(0, 1, 0)    
    else:
        cr.set_source_rgb(1, 1, 1)
def main():
    f=gl.glob("/home/fast/onimaru/data/reconstruction/network_constructor_deepsea_1d3_Sat_Jul__1_145520_2017.ckpt-6293_transpose_*.npz")
    for npz in f:
        with np.load(npz, 'r') as f:
            
            reconstruct=f["conv2"]
            original_seq=f["original"]
        i=0 
        j=0
        k=0
        l=0
        reconstruct=np.reshape(reconstruct, (1000, 4))
        original_seq=np.reshape(original_seq, (1000, 4))
        
        line_num=10
        DNA_len=1000
        width=DNA_len*30/line_num+200
        hight=1024*2*3
        y_center=300
        ims1 = cairo.PDFSurface(npz+".pdf", width, hight)       
        cr = cairo.Context(ims1)
        cr.move_to(100, y_center)
        cr.line_to(DNA_len/line_num*30+100, y_center)
        #cr.move_to(50, 100)
        #cr.line_to(50, 412)
        cr.set_line_width(2)
        cr.stroke()
        max_value=reconstruct.max()
        SCALE=300/max_value
        for k in range(1000):
            if not k==0 and k%(DNA_len/line_num)==0:
                cr.set_source_rgba(0.0,0.0,0,1.0)
                y_center+=400
                cr.move_to(100, y_center)
                cr.line_to(DNA_len//line_num*30+100, y_center)
                cr.stroke()
                print(y_center)
            max_value=np.amax(reconstruct[k])
            sum_value=np.sum(reconstruct[k])
            max_value2=np.amax(original_seq[k])
            print(max_value)
            if max_value>0.0:
                max_index=np.argmax(reconstruct[k])
            
                if max_index==0:
                    Nuc="A"
                elif max_index==1:
                    Nuc="G"
                elif max_index==2:
                    Nuc="C"
                elif max_index==3:
                    Nuc="T"
            else:
                Nuc="N"
                
            if max_value2>0.0:
                max_index2=np.argmax(original_seq[k])
            
                if max_index2==0:
                    Nuc2="A"
                elif max_index2==1:
                    Nuc2="G"
                elif max_index2==2:
                    Nuc2="C"
                elif max_index2==3:
                    Nuc2="T"
            else:
                Nuc2="N"
                
            
            Nucpos=0
            Nucneg=0
            Nucsize=max_value*SCALE
            Nucsize2=sum_value*SCALE
            x_pos=k%(DNA_len/line_num)
            #cr.move_to(50+x_pos*40*0.75, y_center)               
            #cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
            #select_color(cr, Nuc)
            #font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=Nucsize+0.1,x0=0.0,y0=0.0)
            #cr.set_font_matrix(font_mat)
            #print Nuc
            #cr.show_text(str(Nuc))
            cr.move_to(100+x_pos*40*0.75, y_center)               
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
            select_color(cr, Nuc2)
            font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=Nucsize+20.0,x0=0.0,y0=0.0)
            cr.set_font_matrix(font_mat)
            cr.show_text(str(Nuc2))    
        #cr.set_font_size(40)
        cr.show_page()
                    
if __name__ == "__main__":    
    main()