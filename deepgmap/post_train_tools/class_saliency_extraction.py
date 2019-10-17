import sys
import tensorflow as tf
import numpy as np
import time
import math
import os
import getopt
from glob import glob 
from natsort import natsorted
import pyBigWig as pbw
import subprocess as subp


PATH_SEP=os.path.sep


def longest_common_prefix(seq1, seq2):
    start = 0
    while start < min(len(seq1), len(seq2)):
        if seq1[start] != seq2[start]:
            break
        start += 1
    return seq1[:start]
def longest_common_suffix(seq1, seq2):
    return longest_common_prefix(seq1[::-1], seq2[::-1])[::-1]

start=time.time()

def genome_scan(filename):
    with open(filename, 'r') as f1:
        file_name=f1.name
        path_sep=os.path.sep
        file_name1=file_name.split(path_sep)
        file_name2=file_name1[-1].split('_')
        chromosome=file_name2[2]
        a=file_name2[3]
        b=a.split('.')
        chr_position=int(b[0])
        #window_id=(file_name2[3])[:3]
        genome_seq=np.load(f1)
        shape_of_genome=genome_seq['genome'].shape
        genome_seq_re=np.reshape(genome_seq['genome'], (shape_of_genome[0], shape_of_genome[1], 4, 1))
        genome_seq_re_list=np.array_split(genome_seq_re, 100)
    return genome_seq_re_list, chromosome, chr_position #, window_id

def run(args=None):
    a=time.asctime()
    b=a.replace(':', '')
    start_at=b.replace(' ', '_')
    BATCH_SIZE=1000
    prefix="class_saliency_map"
    GPUID="0"
    genome_file=""
    class_of_interest=-1
    if args==None:
        try:
            options, args =getopt.getopt(sys.argv[1:], 'l:t:o:c:G:g:p:', 
                                         ['log=','test_genome=','output_dir=',"class_of_interest=", "GPUID=", "genoem_file=","prefix="])
        except getopt.GetoptError as err:
            print(str(err))
            sys.exit(2)
        if len(options)<3:
            print('too few argument')
            sys.exit(0)
            
        for opt, arg in options:
            if opt in ('-l', '--log'):
                log_file_name=arg
            elif opt in ('-t', '--test_genome'):
                test_genome=arg
            elif opt in ('-o', '--output_dir'):
                output_dir=arg
                if not output_dir.endswith("/"):
                    output_dir+="/"
            elif opt in ('-p','--prefix'):
                prefix=arg
            elif opt in ('-c', "--class_of_interest"):
                class_of_interest=int(arg)
            elif opt in ('-G', "--GPUID"):
                GPUID=arg
            elif opt in ('-g', '--genome_file'):
                genome_file=arg
    else:
        log_file_name=args.log
        test_genome=args.test_genome
        output_dir=args.output_dir
        if not output_dir.endswith(PATH_SEP):
            output_dir+=PATH_SEP
        prefix=args.prefix
        class_of_interest=args.class_of_interest
        GPUID=args.GPUID
        genome_file=args.genome_file
    chromosome_sizes={}
    with open(genome_file, "r") as fin:
        for line in fin:
            line=line.split()
            chromosome_sizes[line[0]]=int(line[1])
    
    input_file_prefix= os.path.splitext(log_file_name)[0]
    current_variable=np.load(input_file_prefix+"_trained_variables.npz")
    
    with open(log_file_name, 'r') as fin:
        for line in fin:
            if line.startswith('<tf.Variable'):
                line=line.split(' ')
                print(line)
                if line[1]=="'prediction/W_fc1:0'":
                    line=line[2].split('=(')[1].strip(",'")
                    first_fc_shape=int(line)
            elif line.startswith("data"):
                line=line.split(':')[1]
                data_length=int(line)
            elif line.startswith("Total class number:"):
                class_num=int(line.split(': ')[1])
            elif line.startswith("Labeled file:"):
                sample_list=[] 
                line=line.split(": ")[1].strip("\n")
                #print line
                if os.path.isfile(line):
                    label_file=line
                else:
                    line=line.strip(".")
                    cwd = os.getcwd()
                    label_file=cwd+line
                if not os.path.isfile(label_file):
                    sys.exit("cannot find "+line)
                with open(label_file) as fin:
                    line2=fin.next()
                    line2=line2.split()[1:]
                    common_prefix = longest_common_prefix(line2[0],line2[-1])
                    common_suffix = longest_common_suffix(line2[0],line2[-1])
                    #print common_prefix, common_suffix
                    common_prefix_len=len(common_prefix)
                    common_suffix_len=len(common_suffix)
                    for l in line2:
                        l=l[common_prefix_len:]
                        l=l[:-common_suffix_len]
                        sample_list.append(l)
                    print(sample_list)
                    
    if not "*" in test_genome:
        test_genome+=PATH_SEP+"*npz"
    test_genome_list=natsorted(glob(test_genome))
    if len(test_genome_list)==0:
        sys.exit(test_genome+" does not exist.")
    
    def recon_variable(shape, variable_name):
        initial = tf.truncated_normal(shape, mean=0.02, stddev=0.02)
        return tf.Variable(initial, name=variable_name, trainable=True)
    def conv2d_1(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 1, 1], padding='VALID')      
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
    def max_pool_4x1(x):
        return tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
    def max_pool_2x2_with_argmax(x):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
    def max_pool_4x1_with_argmax(x):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')    
    
    
    
    
    #x_image_recon = recon_variable([1, 1000, 4, 1], 'x_image_recon')
    #print GPUID
    
    if class_of_interest==-1:
        classes =range(class_num)
    else:
        classes=[class_of_interest]
    
    for c in classes:
        _tmp_prefix=prefix+"_class_"+sample_list[c]
        out_pf=output_dir+_tmp_prefix
        
        if os.path.isfile(out_pf+".bw"):
            print("skipping "+prefix+"_class_"+sample_list[c])
            continue
        """elif c<=18:
            print("skipping "+prefix+"_class_"+sample_list[c])
            continue"""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess2 = tf.Session(config=config)
        
        
        with tf.device('/device:GPU:'+GPUID):
            x_image_recon=tf.placeholder(tf.float32, shape=[BATCH_SIZE, data_length, 4, 1])
            with tf.GradientTape() as g:
                g.watch(x_image_recon)
                h_conv11_re=conv2d_1(x_image_recon, current_variable["prediction/W_conv1:0"])
                h_conv12_re=conv2d_1(x_image_recon, tf.reverse(current_variable["prediction/W_conv1:0"], [0,1]))
                h_conv11_re_ = tf.nn.relu(h_conv11_re)
                h_conv12_re_ = tf.nn.relu(h_conv12_re)
                h_pool1_re = max_pool_2x2(h_conv11_re_)
                h_pool1_rc_re = max_pool_2x2(h_conv12_re_)
                h_conv2_re = tf.add(tf.nn.relu(conv2d_1(h_pool1_re, current_variable["prediction/W_conv2:0"])), tf.nn.relu(conv2d_1(h_pool1_rc_re, tf.reverse(current_variable["prediction/W_conv2:0"], [0,1]))))
                h_pool2_re = max_pool_2x2(h_conv2_re)
                h_conv21_re = tf.nn.relu(conv2d_1(h_pool2_re, current_variable["prediction/W_conv21:0"]))
                h_pool21_re = max_pool_2x2(h_conv21_re)
                h_conv22_re = tf.nn.relu(conv2d_1(h_pool21_re, current_variable["prediction/W_conv22:0"]))
                h_pool22_re = max_pool_4x1(h_conv22_re)
                
                h_pool3_flat_re = tf.reshape(h_pool22_re, [-1, 1*first_fc_shape])
                
                h_fc1_re = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat_re, current_variable["prediction/W_fc1:0"]), current_variable["prediction/b_fc1:0"]))
                y_conv_re=tf.add(tf.matmul(h_fc1_re,current_variable["prediction/W_fc4:0"]), current_variable["prediction/b_fc4:0"])
                #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][0])/(tf.nn.sigmoid(y_conv_re[0][2])+tf.nn.sigmoid(y_conv_re[0][0])+tf.nn.sigmoid(y_conv_re[0][1])+0.000001),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
                #print y_conv_re.shape
                #cost =tf.nn.sigmoid(y_conv_re[:,class_of_interest])
                #cost =tf.nn.relu(y_conv_re[:,class_of_interest])
                cost =tf.clip_by_value(y_conv_re[:,c], -4.0, 1000000000.0)
                #cost =y_conv_re[:,class_of_interest]
                #print cost.shape
            w=g.gradient(cost, x_image_recon)
        
        sess2.run(tf.global_variables_initializer())
        position_list=[]
        sal_map=[]
        BREAK=False
    
        i=0
        
        chrom_list=[]
        start_list=[]
        end_list=[]
        value_list=[]
        chrom_set=[]
        header_list=[]
        bw=None    
        bw_file_list=[]
        bw_file_merged_list=[]
        for test_genome_ in test_genome_list:
            print("reading "+test_genome_)
            genome_data=np.load(test_genome_)
            position_list_, seq_list=genome_data['positions'], genome_data['sequences']
            
            
            seq_list=np.array(seq_list, np.int16).reshape(-1, data_length, 4, 1)
            seq_length=seq_list.shape[0]
            #print(seq_length)
            
            
            loop=int(math.ceil(float(seq_length)/BATCH_SIZE))
            for i in range(loop):
                if i*BATCH_SIZE>seq_length:
                    break
                scanning=seq_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                position=position_list_[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                len_scanning=len(scanning)
                if len_scanning<BATCH_SIZE:
                    dummy_array=np.zeros([(BATCH_SIZE-len_scanning), data_length, 4, 1])
                    scanning=np.concatenate([scanning, dummy_array])
                    
                w_tmp=sess2.run(w, feed_dict={x_image_recon: scanning})
                #print w_tmp.shape
                #print w_tmp[1]
                w_tmp_shape=w_tmp.shape
                #print w_tmp[0]
                w_tmp=np.reshape(w_tmp,[w_tmp_shape[0], w_tmp_shape[1],w_tmp_shape[2]])
                #w_tmp=np.amax(np.absolute(np.clip(w_tmp, None, 0.0)), axis=2)
                w_tmp=np.sum(np.absolute(w_tmp), axis=2)
    
                if len_scanning<BATCH_SIZE:
                    w_tmp=w_tmp[:len_scanning]
                for j in range(len_scanning):
                    
                    sal_map=np.reshape(w_tmp[j], [-1])
                    #print np.sum(sal_map)
                    if not np.sum(sal_map)==0:
                        #print position[j]
                        current_chr, current_pos=position[j].strip(">").split(':')
                        if not current_chr in chrom_set:
                            chrom_set.append(current_chr)
                            if bw is not None:
                                bw.close()
                                if len(bw_file_list)==50:
                                    k=len(bw_file_merged_list)
                                    tmp_name=out_pf+"_"+start_at+"tmp_"+str(k)
                                    subp.check_call(["bigWigMerge"]+bw_file_list+[ tmp_name+".bedGraph"])
                                    with open(tmp_name+".bedGraph" ,"w") as tmp_out:
                                        subp.check_call(["sort", "-k1,1", "-k2,2n",tmp_name+".bedGraph"], stdout=tmp_out)
                                    subp.check_call(["bedGraphToBigWig", tmp_name+".bedGraph", genome_file,tmp_name+".bw"])
                                    bw_file_merged_list.append(tmp_name+".bw")
                                    subp.check_call(["rm", tmp_name+".bedGraph"])
                                    for _file in bw_file_list:
                                        subp.check_call(["rm", _file])
                                    bw_file_list=[]
                            bw=pbw.open(out_pf+"_"+current_chr+".bw", "w")
                            print("writing "+out_pf+"_"+current_chr+".bw")
                            bw_file_list.append(out_pf+"_"+current_chr+".bw")
                            bw.addHeader([(current_chr, chromosome_sizes[current_chr])])
                            
                            
                        start, end =map(int, current_pos.split("-"))
                        #print current_chr, len(sal_map), start,end
                        bw.addEntries([current_chr]*len(sal_map), 
                                  range(start,end), 
                                  ends=range(start+1,end+1), 
                                  values=sal_map)
                
                """if len(sal_map)==0:
                    sal_map=np.reshape(w_tmp, [-1])
                else:
                    sal_map=np.concatenate([sal_map,np.reshape(w_tmp, [-1])],axis=0)
                """
                
                if len(bw_file_merged_list)==50:
                    BREAK=True
                    break
            if BREAK:
                break
        bw.close()
        
        
        if len(bw_file_merged_list)==0:
            #h, t=os.path.split(bw_file_list[0])
            if len(bw_file_list)==1:
                subp.check_call(["mv", bw_file_list[0], out_pf+".bw"])
            else:
                tmp_name=out_pf+"_"+start_at+"_tmp"
                print("merging files to create "+out_pf+".bw")
                #print bw_file_list
                subp.check_call(["bigWigMerge"]+bw_file_list+[tmp_name+".bedGraph"])
                with open(out_pf+".bedGraph" ,"w") as tmp_out:
                    subp.check_call(["sort", "-k1,1", "-k2,2n",tmp_name+".bedGraph"], stdout=tmp_out)
                
                subp.check_call(["bedGraphToBigWig", out_pf+".bedGraph", genome_file, out_pf+".bw"])
                subp.check_call(["rm", out_pf+".bedGraph"])
                subp.check_call(["rm", tmp_name+".bedGraph"])
                for _file in bw_file_list:
                    subp.check_call(["rm", _file])
        else:
            if len(bw_file_list)>1:
                #print bw_file_list
                k=len(bw_file_merged_list)
                tmp_name=out_pf+"_"+start_at+"tmp_"+str(k)
                k=len(bw_file_merged_list)
                
                subp.check_call(["bigWigMerge"]+bw_file_list+[tmp_name+".bedGraph"])
                with open(out_pf+".bedGraph" ,"w") as tmp_out:
                    subp.check_call(["sort", "-k1,1", "-k2,2n",tmp_name+".bedGraph"], stdout=tmp_out)
                subp.check_call(["bedGraphToBigWig", tmp_name+".bedGraph", genome_file, tmp_name+".bw"])
                bw_file_merged_list.append(tmp_name+".bw")
                subp.check_call(["rm", tmp_name+".bedGraph"])
                #subp.check_call(["rm", "_tmp.bedGraph"])
                for _file in bw_file_list:
                    subp.check_call(["rm", _file])
                bw_file_list=[]
            if len(bw_file_list)>0:
                bw_file_merged_list.append(bw_file_list[0])
            
            tmp_name=out_pf+"_"+start_at+"_tmp"
            subp.check_call(["bigWigMerge"]+bw_file_merged_list+[tmp_name+".bedGraph"])
            with open(out_pf+".bedGraph" ,"w") as tmp_out:
                subp.check_call(["sort", "-k1,1", "-k2,2n",tmp_name+".bedGraph"], stdout=tmp_out)
            subp.check_call(["rm", tmp_name+".bedGraph"])
            
            subp.check_call(["bedGraphToBigWig", out_pf+".bedGraph", genome_file, out_pf+".bw"])
            subp.check_call(["rm", out_pf+".bedGraph"])
            
            for _file in bw_file_merged_list:
                    subp.check_call(["rm", _file])

if __name__== '__main__':
    run()