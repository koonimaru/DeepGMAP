import tensorflow as tf
import sys
import numpy as np
import time
import glob
from natsort import natsorted
import getopt
import importlib as il
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde

PATH_SEP=os.path.sep

def next_batch(loop, input_dir, batch_size, data_length):
    f = glob.glob(str(input_dir)+"*")
    f_srt=natsorted(f)
    with np.load(str(f_srt[loop])) as f1:
        try:
            dnase_data_labels=f1['labels'], f1['data_array']
            
        except EOFError:
            print("cannot load: "+str(f_srt[loop]))
        images=np.reshape(dnase_data_labels[1], (batch_size, data_length, 4, 1))
        labels=dnase_data_labels[0]
        halfimages=np.vsplit(images, 2)
        halflabels=np.vsplit(labels, 2)
    return halfimages[0], halflabels[0], halfimages[1], halflabels[1]

def process(f,half_batch,data_length):
    with np.load(f) as f1:
        try:
            labels, images=f1['labels'], f1['data_array']
            
        except EOFError:
            print("cannot load: "+str(f))
    
    shape=images.shape
    if len(shape)<3:
        print(f)
        return [], []
    
    images=np.reshape(images, (shape[0], shape[1], shape[2], 1))
    
    #print(shape[0])          
    if shape[0]>half_batch:
        halfimages=images[:half_batch] , images[half_batch:]
        halflabels=labels[:half_batch], labels[half_batch:]
    else:
        halfimages=images
        halflabels=labels
    
    return halfimages, halflabels

def process2(f,data_length):
    with np.load(f) as f1:
        try:
            dnase_data_labels=f1['labels'], f1['data_array']
            
        except EOFError:
            print("cannot load: "+str(f))
    
    shape=dnase_data_labels[1].shape
    images=np.reshape(dnase_data_labels[1], (shape[0], shape[1], shape[2], 1))
    labels=dnase_data_labels[0]      
    return images, labels


def batch_queuing(file_list, batch_size, data_length):
    
    with tf.device('/cpu:0'):
        half_batch=batch_size//2
        image_list=[]
        label_list=[]
        #CPU=20
        #pool=mltp.Pool(CPU)
        for f in file_list:
            #res=apply_async(pool, process,args=(f,))
            #halfimages, halflabels=res.get()
            
            halfimages, halflabels=process(f,half_batch,data_length)
            if len(halfimages)==0:
                break
            image_list.append(halfimages)
            label_list.append(halflabels)
        #pool.close()
        #pool.join()
        return image_list, label_list

def batch_queuing2(file_list, batch_size, data_length):
    
    with tf.device('/cpu:0'):
        image_list=[]
        label_list=[]
        #CPU=20
        #pool=mltp.Pool(CPU)
        for f in file_list:
            #res=apply_async(pool, process,args=(f,))
            #halfimages, halflabels=res.get()
            
            images, labels=process2(f,data_length)
            image_list.append(images)
            label_list.append(labels)
        #pool.close()
        #pool.join()
        return image_list, label_list


def softmax(w, t = 1.0):
    npa = np.array
    e = np.exp(npa(w) / t)
    dist = e /np.stack((np.sum(e, axis=1),np.sum(e, axis=1)),axis=-1)
    return dist
def test_batch(input_dir,output_dir,test_batch_num,batch_size, data_length):
    f = glob.glob(str(input_dir))
    f_srt=natsorted(f, key=lambda y: y.lower())
    #print len(f_srt), test_batch_num
    data_list=[]
    labels_list=[]
    for i in range(3):
        a=np.load(f_srt[int(test_batch_num)+i])
        label_=a['labels'], 
        data_=a['data_array']
        data_shape=np.shape(data_)
        label_shape=np.shape(label_)
        #print "labelshape="+str(label_shape)
        data_list.append(np.reshape(data_, (data_shape[0], data_length, 4, 1)))
        labels_list.append(np.reshape(label_,(-1,label_shape[-1])))

    return data_list[0], labels_list[0], data_list[1], labels_list[1], data_list[2], labels_list[2]

def div_roundup(x, y):
    if y%x==0:
        return y//x
    else:
        return y//x+1
    
def run(args):
    main(args)


def main(args=None):
    
    start=time.time()
    a=time.asctime()
    b=a.replace(':', '')
    start_at=b.replace(' ', '_')
    mode="train"
    loop_num_=None
    test_batch_num=None
    max_to_keep=2
    TEST_THRESHOLD=0.75
    SAVE_THRESHOLD=0
    dropout_1=1.00
    dropout_2=0.80
    dropout_3=0.50
    queue_len=5000
    EPOCHS_TO_TRAIN=1
    #max_train=20000
    
    if args!=None:
        mode=args.mode
        loop_num_=args.loop_number
        test_batch_num=args.test_batch_number
        max_to_keep=args.max_to_keep
        input_dir=args.in_directory
        model_name=args.model
        pretrained_dir=args.ckpt_file
        output_dir=args.out_directory
        TEST_FREQ=args.test_frequency
        EPOCHS_TO_TRAIN=args.epochs
        GPUID=str(args.gpuid)
        TEST_THRESHOLD=args.test_threshold
    else:
        try:
            options, args =getopt.getopt(sys.argv[1:], 'm:i:n:b:o:c:p:', ['mode=', 'in_dir=', 'loop_num=', 'test_batch_num=', 'out_dir=','network_constructor=','pretrained_model='])
        except getopt.GetoptError as err:
            print(str(err))
            sys.exit(2)
        if len(options)<3:
            print('too few argument')
            sys.exit(0)
        for opt, arg in options:
            if opt in ('-m', '--mode'):
                mode=arg
            elif opt in ('-i', '--in_dir'):
                input_dir=arg
                                    
            elif opt in ('-n', '--loop_num'):
                loop_num_=int(arg)
            elif opt in ('-b', '--test_batch_num'):
                test_batch_num=int(arg)
            elif opt in ('-o', '--out_dir'):
                if not arg.endswith(PATH_SEP):
                    arg+=PATH_SEP
                output_dir=arg
            elif opt in ('-c', '--network_constructor'):
                model_name=arg
            elif opt in ('-p', '--pretrained_model'):
                pretrained_dir=arg
    
    
        
    if input_dir.endswith(PATH_SEP):
        input_dir=str(input_dir)+"*.npz"
    elif input_dir.endswith("*"):
        input_dir=str(input_dir)+".npz"
    elif input_dir.endswith("*.npz"):
        pass
    else:
        input_dir=str(input_dir)+PATH_SEP+"*.npz"
    f = glob.glob(input_dir)
    if len(f)==0:
        print("can't open input files, no such a directory")
        sys.exit(0)
    
    f_srt=natsorted(f)
    
    if loop_num_==None:
        loop_num_=len(f_srt)-5
        
    if test_batch_num==None:
        test_batch_num=loop_num_+1
        
    
    with np.load(str(f_srt[0])) as f:
        labels=f['labels']
        _data=f['data_array']
        batch_size, label_dim=labels.shape
        _, data_length, dna_dim=_data.shape
        print(batch_size, label_dim)    
    """
    if not os.path.exists(output_dir):
        yesno=input(output_dir+" does not exist. Do you want to create a new one? y/n: ")
        if yesno=="y":
            try:
                os.makedirs(output_dir)
            except:
                sys.exit("cannot create the output directory.")
        else:
            sys.exit("please set -o parameter.")
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            sys.exit("cannot create an output directory. Please set a valid output directory with -o option.")
    output_dir+=PATH_SEP
    saving_dir_prefix=str(output_dir)+str(model_name)+"_"+start_at
    if not os.path.exists(saving_dir_prefix):
        os.makedirs(saving_dir_prefix)
    else:
        sys.exit(saving_dir_prefix +" already exists.")
    saving_dir_prefix=saving_dir_prefix+PATH_SEP+"train"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    x_image = tf.placeholder(tf.float32, shape=[None, data_length, dna_dim, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, label_dim])
    phase=tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    keep_prob3 = tf.placeholder(tf.float32)
    nc=il.import_module("deepgmap.network_constructors."+str(model_name))
    print("running "+str(model_name))
    
    model = nc.Model(image=x_image, label=y_, 
                     output_dir=saving_dir_prefix,
                     phase=phase, 
                     start_at=start_at, 
                     keep_prob=keep_prob, 
                     keep_prob2=keep_prob2, 
                     keep_prob3=keep_prob3, 
                     data_length=data_length,
                     max_to_keep=max_to_keep,
                     GPUID=GPUID)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver=model.saver

    if not pretrained_dir is None:
        if not os.path.isfile(pretrained_dir):
            sys.exit(pretrained_dir + ' does not exist.')
        if pretrained_dir.endswith(".meta"):
            pretrained_dir=os.path.splitext(pretrained_dir)[0]
        saver.restore(sess, pretrained_dir)
        print("pre-trained variables have been restored.")
    
    train_accuracy_record=[]
    loss_val_record=[]
    total_learing=[]
    loop_num=div_roundup(queue_len, len(f_srt))
    BREAK=False
    prev_ac=None
    test_step=[]
    CHECK_TEST_FR=False
    computation_time=[]
    t_batch = test_batch(input_dir,output_dir,test_batch_num,batch_size, data_length)
    if label_dim>100:
        TEST_FREQ=20
    h=0
    for _ in range(EPOCHS_TO_TRAIN):
        if BREAK:
                break
        for i in range(loop_num):
            if BREAK:
                print("breaking the train loop")
                break
            input_files=f_srt[i*queue_len:(i+1)*queue_len]
            image_list, label_list=batch_queuing(input_files, batch_size, data_length)
            
            for k in range(len(image_list)):
                
                a=np.shape(image_list[k])
                if not len(a)==4:
                    batch=image_list[k][0],label_list[k][0],image_list[k][1],label_list[k][1]
                    
                if k%TEST_FREQ==0:
                    
                    #print a
                    if len(a)==4:
                        train_accuracy_,loss_val= sess.run([model.error, 
                                                            model.cost], 
                                                            feed_dict=
                                                            {x_image: image_list[k], 
                                                            y_: label_list[k], 
                                                            keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, 
                                                            phase: False})
                    else:
                        
                        #print(len(batch))
                        #batch = next_batch(i,input_files, batch_size, data_length)
                        """train_accuracy_,loss_val= sess.run([model.error, model.cost], 
                                                           feed_dict={x_image: np.concatenate((batch[2],batch[0])), 
                                                                       y_: np.concatenate((batch[3],batch[1])), 
                                                                       keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, 
                                                                       phase: False})"""
                        train_accuracy_,loss_val= sess.run([model.error, model.cost], feed_dict={x_image:batch[2], 
                                                                                                 y_: batch[3], 
                                                                                                keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, 
                                                                                                phase: False})
                    FPR_list, TPR_list, PPV_list=train_accuracy_
                    #print np.nansum(PPV_list)
                    curr_accu=float(np.round(np.nanmean(2*np.array(TPR_list)*np.array(PPV_list)/(0.0000001+np.array(PPV_list)+np.array(TPR_list))),4))
                    sys.stdout.write("\r"+"step "+str(h)
                          +", cost: "+str(loss_val)
                          +", train_accuracy: "
                          +str(curr_accu)+"                ")            
                    sys.stdout.flush()
                    
                    #train_accuracy_record.append(TPR_list[0]-FPR_list[0])
                    train_accuracy_record.append(curr_accu)
                    loss_val_record.append(loss_val)
                    total_learing.append((h)*batch_size/1000.0)
                    if len(train_accuracy_record)>=3:
                        #temporal_accuracy=train_accuracy_record[i*queue_len+k]+train_accuracy_record[i*queue_len+k-1]+train_accuracy_record[i*queue_len+k-2]
                        temporal_accuracy=np.round((train_accuracy_record[-1]+train_accuracy_record[-2]+train_accuracy_record[-3])/3.0,4)
                        if len(test_step)>1:
                            CHECK_TEST_FR=((h-test_step[-1])>1000)
                        CHECK_ACCU=(temporal_accuracy>=TEST_THRESHOLD)
                        if CHECK_ACCU or CHECK_TEST_FR:
                            
                            test_step.append(h)
                            if len(test_step)>10:
                                e, f=test_step[-1]//TEST_FREQ,test_step[-10]//TEST_FREQ
                                if e-f<=40:
                                    TEST_THRESHOLD+=0.02
                                    
                                    if TEST_THRESHOLD>0.9900:
                                        TEST_THRESHHOLD=0.9900
                                    print("\n"+str(TEST_THRESHOLD))
                            if CHECK_TEST_FR:
                                TEST_THRESHOLD-=0.02
                            #TEST_THRESHHOLD=temporal_accuracy-0.005
                            
                            
                            
                            f1_list=[]
                            for o in range(3):
                                ta=sess.run(model.error, feed_dict={x_image: t_batch[o*2], y_: t_batch[o*2+1], keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
                                FPR_list, TPR_list, PPV_list=ta
                                
                                f1=float(np.round(np.nanmean(2*np.array(TPR_list)*np.array(PPV_list)/(0.0000001+np.array(PPV_list)+np.array(TPR_list))),4))
                                f1_list.append(f1)                    
        
        
                            mean_ac=np.round(np.nanmean(f1_list),2)
                            to_print=("\nThis is tests for the model at the train step: "+str(i*queue_len+k)+"\n"
                                      +"mean accuracy : "+str(mean_ac)
                                      +"\n Total time "+ str(time.time()-start))
                            print(to_print)
                            if (prev_ac==None and mean_ac>=SAVE_THRESHOLD) or (prev_ac!=None and mean_ac>=prev_ac):
                                
                                flog=open(saving_dir_prefix+'.log', 'a')
                                flog.write("This is tests for the model at the train step: "+str(h)+"\nThe average of F1: "+str(mean_ac)+'\n')
                                flog.close()
                                saver.save(sess, saving_dir_prefix+'.ckpt', global_step=h)
                                prev_ac=mean_ac                    
                                
                            if mean_ac>=0.999:
                                BREAK=True
                                break
                #sess.run(model.optimize, feed_dict={x_image: np.concatenate((batch[2],batch[0])),y_: np.concatenate((batch[3],batch[1])), keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                if len(a)==4:
                    start_tmp=time.time()
                    sess.run(model.optimize, feed_dict={x_image: image_list[k], y_:label_list[k], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                    computation_time.append(time.time()-start_tmp)
                else:
                    start_tmp=time.time()
                    sess.run(model.optimize, feed_dict={x_image: batch[2], y_: batch[3], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                    
                    sess.run(model.optimize, feed_dict={x_image: batch[0], y_: batch[1], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                    sess.run(model.optimize, feed_dict={x_image: batch[2], y_: batch[3], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                    sess.run(model.optimize, feed_dict={x_image: batch[0], y_: batch[1], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                    computation_time.append(time.time()-start_tmp)
                if len(computation_time)==24:
                    print("\nMean computation time per mini-batch is so far: "+str(np.mean(computation_time[3:])))
                if TEST_FREQ>=50:
                    sys.stdout.write("\r"+"step: "+str(h))            
                    sys.stdout.flush()
                
                h+=1
                if h>=loop_num_*EPOCHS_TO_TRAIN: # or (i*queue_len+k) >= max_train:
                    BREAK=True
                    break
    saver.save(sess, saving_dir_prefix+".ckpt", global_step=h)
    
    f1_list=[]
    for o in range(3):
        ta=sess.run(model.error, feed_dict={x_image: t_batch[o*2], y_: t_batch[o*2+1], keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
        FPR_list, TPR_list, PPV_list=ta
        
        f1=float(np.round(np.nanmean(2*np.array(TPR_list)*np.array(PPV_list)/(0.0000001+np.array(PPV_list)+np.array(TPR_list))),4))
        #print(f1)
        f1_list.append(f1)
    
    
    current_variable={}
    all_tv=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in all_tv:
        value=sess.run(v)
        scope=v.name
        current_variable[scope]=value

    current_variable["fc1_param"]=model.fc1_param
    all_=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    np.savez(saving_dir_prefix+'_trained_variables.npz', **current_variable)
    #np.savez(str(output_dir)+str(model_name)+'_local_variables_'+str(start_at)+'.npz', **local_variable)
    mean_ac=np.round(np.nanmean(f1_list),4) 
    running_time=time.time()-start
    
    input_log=os.path.split(input_dir)[0]+PATH_SEP+"data_generation.log"
    labeled_file_name=""
    excluded_chr=""
    if os.path.isfile(input_log):
        with open(input_log, 'r') as ilog:
            for line in ilog:
                if line.startswith("Labeled"):
                    labeled_file_name=line.split(":")[1]
                elif line.startswith("Excluded"):
                    excluded_chr=line.split(':')[1].strip('[]\n')
                    
    import datetime
    to_print=("\nDropout parameters: "+str(dropout_1)+", "+str(dropout_2)+", "+str(dropout_3)+"\n"
              +"Input directory: "+str(input_dir)+"\n"
              +"The average F1: "+str(np.round(mean_ac,2))
              +"\nTotal time: "+ str(datetime.timedelta(seconds=running_time))
              +"\nMean time for one batch: "+str(np.mean(computation_time))
              +"\nModel: "+str(model_name)
              +"\nThe last check point: "+saving_dir_prefix+".ckpt-"+str(h)+".meta"
              +"\nArguments: "+" ".join(sys.argv[1:])
              +"\nTotal class number: "+str(label_dim)
              +"\nLabeled file: "+labeled_file_name.strip('\n')
              +"\nExcluded chromosome: "+excluded_chr
              +"\nGlobal variables: "+"\n".join(map(str, all_)))
       
    sess.close()
    print(to_print)
    flog=open(saving_dir_prefix+'.log', 'a')
    flog.write(to_print+'\n')
    flog.close()
    
    #fit=np.polyfit(total_learing, train_accuracy_record, 1)
    #fit_fn=np.poly1d(fit)
    
    plt.figure(1)
    ax1=plt.subplot(211)
    plt.title('Train accuracy')
    #plt.plot(total_learing, train_accuracy_record, 'c.', total_learing, fit_fn(total_learing), 'm-')
    x=np.nan_to_num(np.array(total_learing))
    y=np.nan_to_num(np.array(train_accuracy_record))
    xy = np.nan_to_num(np.vstack([x,y]))
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    ax1.scatter(x, y, c=z, s=10, edgecolor='')
    ax1.grid(True)

    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,1.0))
    
    plt.figure(1)
    plt.subplot(212)
    plt.title('Cost')
    plt.plot(total_learing,loss_val_record, '-')
    
    
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,1.0))
    plt.savefig(saving_dir_prefix+'_plot.pdf', format='pdf')
    np.savez_compressed(saving_dir_prefix+'_train_rec',total_learing=total_learing, train_accuracy_record=train_accuracy_record,loss_val_record=loss_val_record)

    
    #plt.show()   


if __name__ == '__main__':
    main()
  
  
  
  
  