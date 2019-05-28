import tensorflow as tf
import sys
import numpy as np
import time
import glob
from natsort import natsorted
import getopt
import importlib as il
import matplotlib.pyplot as plt

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
            dnase_data_labels=f1['labels'], f1['data_array']
            
        except EOFError:
            print("cannot load: "+str(f))
    
    shape=dnase_data_labels[1].shape
    images=np.reshape(dnase_data_labels[1], (shape[0], data_length, 4, 1))
    labels=dnase_data_labels[0]  
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
    images=np.reshape(dnase_data_labels[1], (shape[0], data_length, 4, 1))
    labels=dnase_data_labels[0]      
    return images, labels


def batch_queuing(file_list, batch_size, data_length):
    
    with tf.device('/cpu:0'):
        half_batch=batch_size/2
        image_list=[]
        label_list=[]
        #CPU=20
        #pool=mltp.Pool(CPU)
        for f in file_list:
            #res=apply_async(pool, process,args=(f,))
            #halfimages, halflabels=res.get()
            
            halfimages, halflabels=process(f,half_batch,data_length)
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
    test_dir=output_dir.replace('output/', '')
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
        return y/x
    else:
        return y/x+1
    
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
    TEST_THRESHHOLD=0.75
    SAVE_THRESHHOLD=0
    dropout_1=1.00
    dropout_2=0.80
    dropout_3=0.50
    queue_len=5000
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
                output_dir=arg
            elif opt in ('-c', '--network_constructor'):
                model_name=arg
            elif opt in ('-p', '--pretrained_model'):
                pretrained_dir=arg
                
    if input_dir.endswith("/"):
        input_dir=str(input_dir)+"*.npz"
    elif input_dir.endswith("*") or input_dir.endswith(".npz"):
        pass
    else:
        input_dir=str(input_dir)+"/*.npz"
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
        _, data_length, _2=_data.shape
        print(batch_size, label_dim)    

    config = tf.ConfigProto(device_count = {'GPU': 2})
    config.gpu_options.allow_growth=True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    x_image = tf.placeholder(tf.float32, shape=[None, data_length, 4, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, label_dim])
    phase=tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    keep_prob3 = tf.placeholder(tf.float32)
    nc=il.import_module("deepgmap.network_constructors."+str(model_name))
    print("running "+str(model_name))

    model = nc.Model(image=x_image, label=y_, 
                     output_dir=output_dir,
                     phase=phase, 
                     start_at=start_at, 
                     keep_prob=keep_prob, 
                     keep_prob2=keep_prob2, 
                     keep_prob3=keep_prob3, 
                     data_length=data_length,
                     max_to_keep=max_to_keep)

    sess.run(tf.global_variables_initializer())
    saver=model.saver

    if mode=='retrain':
        saver.restore(sess, pretrained_dir)
    
    train_accuracy_record=[]
    loss_val_record=[]
    total_learing=[]
    loop_num=div_roundup(queue_len, len(f_srt))
    BREAK=False
    prev_ac=None
    test_step=[]
    CHECK_TEST_FR=False
    for i in range(loop_num):
        if BREAK:
            print("breaking the train loop")
            break
        input_files=f_srt[i*queue_len:(i+1)*queue_len]
        image_list, label_list=batch_queuing(input_files, batch_size, data_length)
        
        for k in range(len(image_list)):
            start_tmp=time.time()
            a=np.shape(image_list[k])

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
                batch=image_list[k][0],label_list[k][0],image_list[k][1],label_list[k][1]
                #print(len(batch))
                #batch = next_batch(i,input_files, batch_size, data_length)
                train_accuracy_,loss_val= sess.run([model.error, model.cost], feed_dict={x_image: np.concatenate((batch[2],batch[0])), 
                                                                                                                   y_: np.concatenate((batch[3],batch[1])), 
                                                                                                                   keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, 
                                                                                                                   phase: False})
                """train_accuracy_,loss_val= sess.run([model.error, model.cost], feed_dict={x_image:batch[2], 
                                                                                         y_: batch[3], 
                                                                                        keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, 
                                                                                        phase: False})"""
            FPR_list, TPR_list, PPV_list=train_accuracy_
            #print np.nansum(PPV_list)
            curr_accu=float(np.round(np.nanmean(2*np.array(TPR_list)*np.array(PPV_list)/(0.0000001+np.array(PPV_list)+np.array(TPR_list))),4))
            sys.stdout.write("\r"+"step "+str(i*queue_len+k)
                  +", cost: "+str(loss_val)
                  +", train_accuracy: "
                  +str(list([curr_accu]))+", "+str(time.time()-start_tmp))            
            sys.stdout.flush()
            
            #train_accuracy_record.append(TPR_list[0]-FPR_list[0])
            train_accuracy_record.append(curr_accu)
            loss_val_record.append(loss_val)
            total_learing.append((i*queue_len+k)*batch_size/1000.0)
            if i*queue_len+k>=2:
                #temporal_accuracy=train_accuracy_record[i*queue_len+k]+train_accuracy_record[i*queue_len+k-1]+train_accuracy_record[i*queue_len+k-2]
                temporal_accuracy=np.round((train_accuracy_record[i*queue_len+k]+train_accuracy_record[i*queue_len+k-1]+train_accuracy_record[i*queue_len+k-2])/3.0,4)
                if len(test_step)>1:
                    CHECK_TEST_FR=((i*queue_len+k-test_step[-1])>1000)
                CHECK_ACCU=(temporal_accuracy>=TEST_THRESHHOLD)
                if CHECK_ACCU or CHECK_TEST_FR:
                    
                    test_step.append(i*queue_len+k)
                    if len(test_step)>10:
                        e, f=test_step[-1],test_step[-10]
                        if e-f<=40:
                            TEST_THRESHHOLD+=0.10
                            print("\n"+str(TEST_THRESHHOLD))
                            if TEST_THRESHHOLD>0.9800:
                                TEST_THRESHHOLD=0.9800
                                
                    if CHECK_TEST_FR:
                        TEST_THRESHHOLD-=0.02
                    #TEST_THRESHHOLD=temporal_accuracy-0.005
                    t_batch = test_batch(input_dir,output_dir,test_batch_num,batch_size, data_length)
                    
                    
                    f1_list=[]
                    for o in range(3):
                        ta=sess.run(model.error, feed_dict={x_image: t_batch[o*2], y_: t_batch[o*2+1], keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
                        FPR_list, TPR_list, PPV_list=ta
                        
                        f1=float(np.round(np.nanmean(2*np.array(TPR_list)*np.array(PPV_list)/(0.0000001+np.array(PPV_list)+np.array(TPR_list))),4))
                        f1_list.append(f1)                    


                    mean_ac=np.round(np.nanmean(f1_list),4)
                    to_print=("\nThis is tests for the model at the train step: "+str(i*queue_len+k)+"\n"
                              +"mean accuracy : "+str(mean_ac)
                              +"\n Total time "+ str(time.time()-start))
                    print(to_print)
                    if (prev_ac==None and mean_ac>=SAVE_THRESHHOLD) or (prev_ac!=None and mean_ac>=prev_ac):
                        
                        flog=open(str(output_dir)+str(start_at)+'.log', 'a')
                        flog.write("This is tests for the model at the train step: "+str(i*queue_len+k)+"\nThe average of TPR+PPV: "+str(mean_ac)+'\n')
                        flog.close()
                        saver.save(sess, str(output_dir)+str(model_name)+"_"+str(start_at)+'_step'+str(i*queue_len+k)+'.ckpt', global_step=i*queue_len+k)
                        prev_ac=mean_ac                    
                        
                    if mean_ac>=0.999:
                        BREAK=True
                        break
            #sess.run(model.optimize, feed_dict={x_image: np.concatenate((batch[2],batch[0])),y_: np.concatenate((batch[3],batch[1])), keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
            if len(a)==4:
                sess.run(model.optimize, feed_dict={x_image: image_list[k], y_:label_list[k], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
            else:
                sess.run(model.optimize, feed_dict={x_image: batch[2], y_: batch[3], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                sess.run(model.optimize, feed_dict={x_image: batch[0], y_: batch[1], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                sess.run(model.optimize, feed_dict={x_image: batch[2], y_: batch[3], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
                sess.run(model.optimize, feed_dict={x_image: batch[0], y_: batch[1], keep_prob: dropout_1, keep_prob2: dropout_2, keep_prob3: dropout_3,phase:True})
            
            if (i*queue_len+k)==loop_num_: # or (i*queue_len+k) >= max_train:
                BREAK=True
                break
            
    saver.save(sess, str(output_dir)+str(model_name)+"_"+str(start_at)+".ckpt", global_step=i*queue_len+k)
             
    t_batch = test_batch(input_dir,output_dir,test_batch_num,batch_size, data_length)   
    f1_list=[]
    for o in range(3):
        ta=sess.run(model.error, feed_dict={x_image: t_batch[o*2], y_: t_batch[o*2+1], keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
        FPR_list, TPR_list, PPV_list=ta
        
        f1=float(np.round(np.nanmean(2*np.array(TPR_list)*np.array(PPV_list)/(0.0000001+np.array(PPV_list)+np.array(TPR_list))),4))
        print(f1)
        f1_list.append(f1)
    
    
    current_variable={}
    all_tv=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for v in all_tv:
        value=sess.run(v)
        scope=v.name
        current_variable[scope]=value
    all_lv=tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
    local_variable={}
    for v in all_lv:
        value=sess.run(v)
        scope=v.name
        print(scope)
        local_variable[scope]=value
    all_=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    np.savez(str(output_dir)+str(model_name)+'_trained_variables_'+str(start_at)+'.npz', **current_variable)
    np.savez(str(output_dir)+str(model_name)+'_local_variables_'+str(start_at)+'.npz', **local_variable)
    mean_ac=np.round(np.nanmean(f1_list),4) 
    running_time=time.time()-start
    import datetime
    if args is not None:
        _args=args
    else:
        _args=sys.argv
    to_print=("dropout parameters: "+str(dropout_1)+", "+str(dropout_2)+", "+str(dropout_3)+"\n"
              +"input directory: "+str(input_dir)+"\n"
              +"The average of TPR+PPV: "+str(np.round(mean_ac,2))
              +"\nTotal time "+ str(datetime.timedelta(seconds=running_time))
              +"\nThe model is "+str(model_name)
              +"\nArguments are "+str(sys.argv[1:])
              +"\nGlobal variables: "+str(all_))
       
    sess.close()
    print(to_print)
    flog=open(str(output_dir)+str(start_at)+'.log', 'a')
    flog.write(to_print+'\n')
    flog.close()
    
    fit=np.polyfit(total_learing, train_accuracy_record, 1)
    fit_fn=np.poly1d(fit)
    
    plt.figure(1)
    ax1=plt.subplot(211)
    plt.title('Train accuracy')
    plt.plot(total_learing, train_accuracy_record, 'c.', total_learing, fit_fn(total_learing), 'm-')
    ax1.grid(True)

    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,1.0))
    
    plt.figure(1)
    plt.subplot(212)
    plt.title('Cost')
    plt.plot(total_learing,loss_val_record, '-')
    
    
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,0,1.0))
    plt.savefig(str(output_dir)+'plot_'+str(start_at)+'.pdf', format='pdf')
    np.savez_compressed(str(output_dir)+str(model_name)+"_"+str(start_at)+'_train_rec',total_learing=total_learing, train_accuracy_record=train_accuracy_record,loss_val_record=loss_val_record)
    
    plt.show()   


if __name__ == '__main__':
    main()
  
  
  
  
  