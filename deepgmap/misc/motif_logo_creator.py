from cairocffi import cairo
import gzip
import pickle       
import numpy as np

def select_color(cr, DNA):
    if DNA=="A":
        cr.set_source_rgb(1, 0, 0)
    elif DNA=="G":
        cr.set_source_rgb(0, 0, 0)
    elif DNA=="C":
        cr.set_source_rgb(0, 0, 1)
    elif DNA=="T":
        cr.set_source_rgb(0, 1, 0)    
   
def main():
    with gzip.open('/media/koh/HD-PCFU3/mouse/variables_999_Sun_Oct_30_120751_2016.cpickle.gz', 'r') as f:
        variables=pickle.load(f)
        filter1=variables[0]
    
    
    
    i=0 
    j=0
    k=0
    l=0
    
    
    filter_shape=filter1.shape
    width=filter_shape[0]*30+100
    hight=512
    y_center=hight/2
    for i in range(filter_shape[3]):
        ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, hight)
        
        
        cr = cairo.Context(ims)
        cr.move_to(50, y_center)
        cr.line_to(filter_shape[0]*30+50, y_center)
        cr.move_to(50, 100)
        cr.line_to(50, 412)
        cr.set_line_width(2)
        cr.stroke()
        for k in range(filter_shape[0]):

            AGCT={}
            values=[]
            A=["A", filter1[k][0][0][i]*1000.0]
            G=["G",filter1[k][1][0][i]*1000.0]
            C=["C",filter1[k][2][0][i]*1000.0]
            T=["T", filter1[k][3][0][i]*1000.0]
            values=[A,G,C,T]
            pos=filter(lambda x:x[1]>=0,values)
            neg=filter(lambda x:x[1]<0,values)
            pos.sort(key=lambda x:x[1])
            neg.sort(key=lambda x:x[1], reverse=True)
            Nucpos=0
            Nucneg=0
            for l in range(len(pos)):
                Nuc=pos[l][0]
                
                Nucsize=abs(pos[l][1])
                
                cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
                select_color(cr, Nuc)
                font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=Nucsize,x0=0.0,y0=0.0)
                cr.set_font_matrix(font_mat)
                cr.move_to(50+k*40*0.75, y_center-Nucpos*0.75)
                cr.show_text(str(Nuc))
                Nucpos+=abs(pos[l][1])
            l=0
            for l in range(len(neg)):
                Nuc=neg[l][0]
                Nucsize=abs(neg[l][1])
                
                cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
                select_color(cr, Nuc)
                font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=-Nucsize,x0=0.0,y0=0.0)
                cr.set_font_matrix(font_mat)
                cr.move_to(50+k*40*0.75, y_center+(Nucneg)*0.75)
                cr.show_text(str(Nuc))
                Nucneg+=abs(neg[l][1])
            
   
    
    
    #cr.set_font_size(40)

        ims.write_to_png("motif_"+str(i)+".png")
        
        
if __name__ == "__main__":    
    main()