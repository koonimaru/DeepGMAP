
import cairocffi as cairo
import numpy as np

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
def seuquence_visualizer2(npz_file, output_file):

    if type(npz_file)==str:
        with np.load(npz_file) as f:
            reconstruct=f["recon"]
    else:
        reconstruct=npz_file

    i=0 
    j=0
    k=0
    l=0
    line_num=10
    DNA_len=1000
    #max_value=reconstruct.max()
    
    reconstruct=np.reshape(reconstruct, (DNA_len, 4))
    max_value=np.max(reconstruct)
    reconstruct=80*reconstruct/max_value
    #print norm
    """reconstruct_pos=reconstruct.clip(min=0)
    reconstruct_neg=reconstruct.clip(max=0)
    reconstruct_pos_sum=np.sum(reconstruct_pos, axis=1)
    reconstruct_neg_sum=np.sum(reconstruct_neg, axis=1)
    
    max_value=reconstruct_pos_sum.max()
    min_value=reconstruct_neg_sum.min()
    min_value_abs=-min_value
    
    if min_value_abs>max_value:
        max_value=min_value_abs"""
    
    #scale_factor=400/max_value
    
    width=DNA_len*30/line_num+200
    hight=1024*2*3
    y_center=300
    ims1 = cairo.PDFSurface(output_file, width, hight)       
    cr = cairo.Context(ims1)
    cr.move_to(100, y_center)
    cr.line_to(DNA_len/line_num*30+100, y_center)
    #cr.move_to(50, 100)
    #cr.line_to(50, 412)
    cr.set_line_width(2)
    cr.stroke()
    
    meme_fileout=open(output_file+'.meme','w')
    meme_fileout.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n\
Background letter frequencies (from uniform background):\nA 0.2500 C 0.2500 G 0.2500 T 0.2500\n\nMOTIF LONG_MOTIF\n\nletter-probability matrix: alength= 4 w= 1000 nsites= 20 E= 0\n")
        
    for k in range(1000):
        if not k==0 and k%(DNA_len/line_num)==0:
            cr.set_source_rgba(0.0,0.0,0,1.0)
            y_center+=300
            cr.move_to(100, y_center)
            cr.line_to(DNA_len/line_num*30+100, y_center)
            cr.stroke()
            print(y_center)
            
        AGCT={}
        values=[]
        #print reconstruct[k]
        
        #reconstruct[k]=reconstruct[k]*50
        #mean=np.mean(reconstruct[k])
        #stdv=np.std(reconstruct[k])
        
        #reconstruct[k]=(reconstruct[k]-mean)/stdv
        probability=np.round(np.true_divide(np.exp(reconstruct[k]),np.nansum(np.exp(reconstruct[k]))),6)
        probability/=np.nansum(probability)
        to_print=""
        for i in range(4):
            if np.isnan(probability[i]):
                print(k)
                probability[i]=0.0
        
        to_print=str(probability[0])+" "+str(probability[2])+" "+str(probability[1])+" "+str(probability[3])+"\n"
        #print(to_print)
        meme_fileout.write(to_print)
        
        ic=np.nansum(probability*np.log2(probability*4+0.0001))*120
        #print ic
        A=["A", probability[0]*ic]
        G=["G",probability[1]*ic]
        C=["C",probability[2]*ic]
        T=["T", probability[3]*ic]

        """
        A=["A", reconstruct[k][0]*scale_factor]
        G=["G",reconstruct[k][1]*scale_factor]
        C=["C",reconstruct[k][2]*scale_factor]
        T=["T", reconstruct[k][3]*scale_factor]"""
        values=[A,G,C,T]
        pos=filter(lambda x:x[1]>=0,values)
        #neg=filter(lambda x:x[1]<0,values)
        pos.sort(key=lambda x:x[1])
        #neg.sort(key=lambda x:x[1], reverse=True)
        Nucpos=0
        #Nucneg=0
        x_pos=k%(DNA_len/line_num)
        
        for l in range(len(pos)):
            Nuc=pos[l][0]
            
            Nucsize=abs(pos[l][1])+0.1
            cr.move_to(100+x_pos*40*0.75, y_center-Nucpos*0.75)               
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
            _select_color(cr, Nuc)
            font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=Nucsize,x0=0.0,y0=0.0)
            cr.set_font_matrix(font_mat)
            cr.show_text(str(Nuc))
            Nucpos+=abs(pos[l][1])
            
        """
        l=0
        for l in range(len(neg)):
            Nuc=neg[l][0]
            Nucsize=abs(neg[l][1])
            
            cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
            _select_color(cr, Nuc)
            font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=-Nucsize,x0=0.0,y0=0.0)
            cr.set_font_matrix(font_mat)
            cr.move_to(100+x_pos*40*0.75, y_center+(Nucneg)*0.75)
            cr.show_text(str(Nuc))
            Nucneg+=abs(neg[l][1])"""
        """     
        max_value=np.amax(reconstruct[k])
        sum_value=np.sum(reconstruct[k])
        print max_value
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

        Nucpos=0
        Nucneg=0
        Nucsize=max_value*1000
        Nucsize2=sum_value*1000"""

        #cr.move_to(50+x_pos*40*0.75, y_center)               
        #cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
        #select_color(cr, Nuc)
        #font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=Nucsize+0.1,x0=0.0,y0=0.0)
        #cr.set_font_matrix(font_mat)
        #print Nuc
        #cr.show_text(str(Nuc))
        """cr.move_to(50+x_pos*40*0.75, y_center)               
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
        select_color(cr, Nuc2)
        font_mat=cairo.Matrix(xx=40.0,yx=0.0,xy=0.0,yy=Nucsize2+20.0,x0=0.0,y0=0.0)
        cr.set_font_matrix(font_mat)
        cr.show_text(str(Nuc2))    """
    #cr.set_font_size(40)
    meme_fileout.close()
    cr.show_page()

def main():
    npz_file='/home/fast2/onimaru/DeepGMAP-dev/data/reconstructions/conv4frss_Fri_May_11_075425_2018.ckpt-16747Tue_May_15_112518_2018_all_.npz'
    a=npz_file.split('/')[-1]
    a=a.split('.')[0]
    output_file=npz_file+'.pdf'
    seuquence_visualizer2(npz_file, output_file)
if __name__ == "__main__":    
    main()
    