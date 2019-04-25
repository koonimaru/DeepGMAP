import functools
import tensorflow as tf
import math
import sys
from tensorflow.contrib import rnn
_ac=il.import_module("deepgmap.network_constructors.auc_calc") 
ac=_ac.auc_pr

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

class Model(object):
    # parameter lists
    initial_variation=0.001 #standard deviation of initial variables in the convolution filters
    #mini batch size
    dimension1=320 #the number of the convolution filters in the 1st layer

    dimension4=925 #the number of the neurons in each layer of the fully-connected neural network
    conv1_filter=26


    
    train_speed=0.0001

    def __init__(self, *args, **kwargs):
        self.data_length=kwargs["data_length"]
        self.image = kwargs["image"]
        self.label = kwargs["label"]
        self.phase=kwargs["phase"]
        self.keep_prob=kwargs["keep_prob"]
        self.keep_prob2=kwargs["keep_prob2"]
        self.keep_prob3=kwargs["keep_prob3"]
        self.start_at=kwargs["start_at"]
        self.output_dir=kwargs["output_dir"]
        self.max_to_keep=kwargs["max_to_keep"]
        self.GPUID=kwargs["GPUID"]
        self.fc1_param=int(math.ceil((self.data_length-self.conv1_filter+1)/13.0))
        self.prediction
        self.optimize
        self.error
        self.saver
        self.cost
        #print 'Running danq model'
        if self.output_dir is not None:
            flog=open(str(self.output_dir)+'.log', 'w')
            flog.write(str(sys.argv[0])+"\n"
                    +"the filer number of conv1:"+ str(self.dimension1)+"\n"
                      +"the filer size of conv1:"+ str(self.conv1_filter)+"\n"

                      +"the number of neurons in the fully-connected layer:"+ str(self.dimension4)+"\n"
                      +"the standard deviation of initial varialbles:"+ str(self.initial_variation)+"\n"
                      +"train speed:"+ str(self.train_speed)+"\n"
                      +"data length:" + str(self.data_length)+"\n")
            flog.close()
        

    @define_scope
    def prediction(self):
        with tf.device('/device:GPU:'+self.GPUID):
            x_image = self.image
    
            def weight_variable(shape, variable_name):
                initial = tf.truncated_normal(shape, mean=0, stddev=self.initial_variation)
                return tf.Variable(initial, name=variable_name)
              
            def bias_variable(shape, variable_name):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial, name=variable_name)
            
            def bias_variable_high(shape, variable_name, carry_bias=-0.1):
                initial = tf.constant(carry_bias, shape=shape)
                return tf.Variable(initial, name=variable_name)
            
            def conv2d_1(x, W):
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
            def conv2d(x, W):
                return tf.nn.conv2d(x, W, strides=[1, 2, 1, 1], padding='VALID')
            def conv2d_depth(x, W):
                return tf.nn.depthwise_conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
            def max_pool_4x1(x):
                return tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
            def max_pool_13x1(x):
                return tf.nn.max_pool(x, ksize=[1, 13, 1, 1], strides=[1, 13, 1, 1], padding='SAME')
     
            # Network Parameters
            #n_input = 28 # MNIST data input (img shape: 28*28)
            #n_steps = 1000 # timesteps
            n_hidden = 320 # hidden layer num of features
            #n_classes = 10 # MNIST total classes (0-9 digits)
    
            #seq_len = tf.placeholder(tf.int32, [None])
            def BiRNN(x):
            
                # Prepare data shape to match `bidirectional_rnn` function requirements
                # Current data input shape: (batch_size, n_steps, n_input)
                # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
            
                # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
                # Define lstm cells with tensorflow
                # Forward direction cell
                #lstm_fw_cell = rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0), self.keep_prob2) #, use_peepholes=True)
                lstm_fw_cell = rnn.DropoutWrapper(rnn.LSTMBlockCell(n_hidden, forget_bias=1.0), self.keep_prob2)
                # Backward direction cell
                #lstm_bw_cell = rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0), self.keep_prob2, use_peepholes=True)
                lstm_bw_cell = rnn.DropoutWrapper(rnn.LSTMBlockCell(n_hidden, forget_bias=1.0), self.keep_prob2)
                # Get lstm cell output
                """
                try:
                    outputs, _, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                          dtype=tf.float32)
                except Exception: # Old TensorFlow version only returns outputs not states
                    outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,x,
                                                    dtype=tf.float32)"""
                outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,x,
                                                    dtype=tf.float32)
                # Linear activation, using rnn inner loop last output
                #return tf.matmul(outputs[-1], weights['out']) + biases['out']
                return tf.concat(outputs,2)
                """
                outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, tf.unstack(x, axis=2),
                                                          dtype=tf.float32)
    
                                                    
                return outputs"""
            
            
            l2norm_list=[]
            W_conv1 = weight_variable([self.conv1_filter, 4, 1, self.dimension1], 'W_conv1')
            cond=tf.constant(0.9)
            wconv1_l2=tf.reduce_sum(tf.square(W_conv1))
            l2norm_list.append(wconv1_l2)
            W_conv1.assign(tf.cond(wconv1_l2>cond, lambda: tf.multiply(W_conv1, cond/wconv1_l2),lambda: W_conv1 ))
            
            h_conv1 = tf.nn.relu(conv2d_1(x_image, W_conv1))
            h_pool1 = tf.nn.dropout(max_pool_13x1(h_conv1), self.keep_prob2)
            print(h_pool1.shape)
            h_pool1_=tf.reshape(h_pool1, [-1, tf.cast(h_pool1.shape[1], tf.int32),tf.cast(h_pool1.shape[3], tf.int32)])
    
            
            pred = BiRNN(h_pool1_)
            
            
            W_fc1 = weight_variable([2*n_hidden*self.fc1_param, self.dimension4], 'W_fc1')
            wfc1_l2=tf.reduce_sum(tf.square(W_fc1))
            l2norm_list.append(wfc1_l2)
            W_fc1.assign(tf.cond(wfc1_l2>cond, lambda: tf.multiply(W_fc1, cond/wfc1_l2),lambda: W_fc1 ))
            b_fc1 = bias_variable([self.dimension4], 'b_fc1')
            bfc1_l2=tf.reduce_sum(tf.square(b_fc1))
            l2norm_list.append(bfc1_l2)
            b_fc1.assign(tf.cond(bfc1_l2>cond, lambda: tf.multiply(b_fc1, cond/bfc1_l2),lambda: b_fc1 ))
            h_pool3_flat = tf.reshape(pred, [-1, 2*n_hidden*self.fc1_param])
            h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, W_fc1), b_fc1))
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
            label_shape=self.label.shape[1]
            W_fc4 = weight_variable([self.dimension4, tf.cast(label_shape, tf.int32)], 'W_fc4')
            wfc4_l2=tf.reduce_sum(tf.square(W_fc4))
            l2norm_list.append(wfc4_l2)
            W_fc4.assign(tf.cond(wfc4_l2>cond, lambda: tf.multiply(W_fc4, cond/wfc4_l2),lambda: W_fc4 ))        
            b_fc4 = bias_variable([label_shape], 'b_fc4')
            bfc4_l2=tf.reduce_sum(tf.square(b_fc4))
            l2norm_list.append(bfc4_l2)
            b_fc4.assign(tf.cond(bfc4_l2>cond, lambda: tf.multiply(b_fc4, cond/bfc4_l2),lambda: b_fc4 ))
            
            
            y_conv=tf.add(tf.matmul(h_fc1_drop, W_fc4), b_fc4)
            
            variable_dict={"W_conv1": W_conv1,  
                            "W_fc1": W_fc1,
                            "b_fc1": b_fc1,
                            "W_fc4": W_fc4,
                            "b_fc4": b_fc4}
            neurons_dict={"h_conv1":h_conv1,"h_fc1": h_fc1}
            
            return y_conv,tf.nn.sigmoid(y_conv), variable_dict, neurons_dict, l2norm_list
    @define_scope
    def saver(self):
        #return tf.train.Saver(var_list=self.prediction[2])
        return tf.train.Saver(max_to_keep=self.max_to_keep)
    @define_scope
    def cost(self):
        with tf.device('/device:GPU:'+self.GPUID):
            """nll=tf.reduce_mean(-tf.reduce_sum(
                tf.log(
                    tf.add(
                        tf.clip_by_value(tf.multiply(self.label, self.prediction[1]),1e-10,1.0),
                        tf.clip_by_value(tf.multiply(tf.subtract(1.00,self.label), tf.subtract(1.00,self.prediction[1])),1e-10,1.0))
                       ), 
                                              reduction_indices=[1]))"""
            nll=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.prediction[0],pos_weight=1.0))
            return nll
            #l2_norm=tf.reduce_sum(self.prediction[4])
            
            #l1_norm=tf.reduce_sum(tf.abs(self.prediction[1]))
            #return tf.add_n([nll,tf.multiply((5*10**-7), l2_norm),tf.multiply((1*10**-8),l1_norm)])
            #return tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(tf.clip_by_value(self.prediction[0],1e-10,1.0))+(1-self.label)*tf.log(tf.clip_by_value(1-self.prediction[0],1e-10,1.0)), reduction_indices=[1]))
    @define_scope
    def optimize(self):
        with tf.device('/device:GPU:'+self.GPUID):
            #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
            #cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_conv*tf.log(tf.clip_by_value(2*y_conv,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))+y_*tf.log(2*tf.clip_by_value(y_,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))), reduction_indices=[1]))
            #cost = tf.reduce_sum(tf.square(y_conv*tf.log(tf.clip_by_value(2*y_conv,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))+y_*tf.log(2*tf.clip_by_value(y_,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))))
            optimizer = tf.train.RMSPropOptimizer(self.train_speed)
    
            return optimizer.minimize(self.cost)

    @define_scope
    def error(self):
        with tf.device('/device:GPU:'+self.GPUID):
            class_n=self.label.shape[1]
            FPR_list=[]
            TPR_list=[]
            PPV_list=[]
            for i in range(class_n):
                
                true=self.label[:,i]
                prob=self.prediction[1][:,i]
                FPR, TPR, PPV=ac(true,prob,0.5)
                FPR_list.append(FPR)
                TPR_list.append(TPR)
                PPV_list.append(PPV)
            
            return FPR_list, TPR_list, PPV_list

    