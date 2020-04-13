import functools
import tensorflow as tf
import math
import sys
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

def _auc_pr(true, prob, threshold):

    pred = tf.where(prob > threshold, tf.ones_like(prob), tf.zeros_like(prob))
    tp = tf.logical_and(tf.cast(pred, tf.bool), tf.cast(true, tf.bool))
    fp = tf.logical_and(tf.cast(pred, tf.bool), tf.logical_not(tf.cast(true, tf.bool)))
    fn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.cast(true, tf.bool))
    tn = tf.logical_and(tf.logical_not(tf.cast(pred, tf.bool)), tf.logical_not(tf.cast(true, tf.bool)))
    FPR = tf.truediv(tf.reduce_sum(tf.cast(fp, tf.int32)),
                     tf.reduce_sum(tf.cast(tf.logical_or(tn, fp), tf.int32)))
    TPR = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)),
                     tf.reduce_sum(tf.cast(tf.logical_or(tp, fn), tf.int32)))
    PPV = tf.truediv(tf.reduce_sum(tf.cast(tp, tf.int32)),
                     tf.reduce_sum(tf.cast(tf.logical_or(tp, fp), tf.int32)))
    return FPR, TPR, PPV


class Model(object):
    # parameter lists
    initial_variation=0.001 #standard deviation of initial variables in the convolution filters
    #mini batch size
    dimension1=300 #the number of the convolution filters in the 1st layer
    dimension2=200 #the number of the convolution filters in the 2nd layer
    dimension21=200
    dimension4=1000 #the number of the neurons in each layer of the fully-connected neural network
    conv1_filter=19
    #conv1_filter2=49
    conv2_filter=11
    conv21_filter=7
    
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
        self.fc1_param=int(math.ceil((math.ceil((math.ceil((
            self.data_length-self.conv1_filter+1)/3.0)
                        -self.conv2_filter+1)/4.0)
                        -self.conv21_filter+1)/4.0))
        self.prediction
        self.optimize
        self.error
        self.saver
        self.cost
        if self.output_dir is not None:
            flog=open(str(self.output_dir)+'.log', 'w')
            flog.write(str(sys.argv[0])+"\n"
                     +"the filer number of conv1:"+ str(self.dimension1)+"\n"
                      +"the filer size of conv1:"+ str(self.conv1_filter)+"\n"
                      +"the filer number of conv2:"+ str(self.dimension2)+"\n"
                      +"the filer size of conv2:"+ str(self.conv2_filter)+"\n"
                      +"the filer number of conv21:"+ str(self.dimension21)+"\n"
                      +"the filer size of conv21:"+ str(self.conv21_filter)+"\n"
                      +"the number of neurons in the fully-connected layer:"+ str(self.dimension4)+"\n"
                      +"the standard deviation of initial varialbles:"+ str(self.initial_variation)+"\n"
                      +"train speed:"+ str(self.train_speed)+"\n"
                      +"data length:" + str(self.data_length)+"\n")
            flog.close()
        

    @define_scope
    def prediction(self):
        with tf.device('/device:GPU:'+self.GPUID):
            x_image = self.image
            phase=self.phase
            
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
            def max_pool_3x1(x):
                return tf.nn.max_pool(x, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME')
            def max_pool_4x1(x):
                return tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
            def max_pool_8x1(x):
                return tf.nn.max_pool(x, ksize=[1, 17, 1, 1], strides=[1, 17, 1, 1], padding='SAME')
            def batch_norm(x,beta,gamma):
                batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
                normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
                return normed 
    
    
            #fc1_param=int(math.ceil((math.ceil((data_length-conv1_filter+1)/4.0)-conv2_filter+1)/2.0))
            #fc1_param=int(math.ceil((math.ceil((math.ceil((math.ceil((data_length-conv1_filter+1)/1.0)-conv2_filter+1)/2.0)-conv22_filter+1)/2.0)-conv3_filter+1)/2.0))   
            
            W_conv1 = weight_variable([self.conv1_filter, 4, 1, self.dimension1], 'W_conv1')
            b_conv1 = bias_variable([self.dimension1], 'b_conv1')
            h_conv1_=conv2d_1(x_image, W_conv1)
            
            #h_conv2 = tf.nn.dropout(tf.nn.relu(tf.nn.batch_normalization(conv2d(h_conv1, W_conv2), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)), keep_prob2)
            #h_conv1 = tf.nn.dropout(tf.nn.relu(batch_norm(h_conv1_,beta1,gamma1, self.dimension2,phase)), self.keep_prob)
            h_conv1 = tf.nn.dropout(tf.nn.relu(h_conv1_), self.keep_prob)       
            h_pool1 = max_pool_3x1(h_conv1)
    
            #beta2 = tf.Variable(tf.constant(0.0, shape=[self.dimension2]),name='beta2', trainable=True)
            #gamma2 = tf.Variable(tf.constant(1.0, shape=[self.dimension2]), name='gamma2', trainable=True)          
            W_conv2 = weight_variable([self.conv2_filter, 1, self.dimension1, self.dimension2], 'W_conv2')
            b_conv2 = bias_variable([self.dimension2], 'b_conv2')
            h_conv2_=conv2d_1(h_pool1, W_conv2)
            #h_conv2 = tf.nn.dropout(tf.nn.relu(batch_norm(h_conv2_,beta2,gamma2, self.dimension2,phase)), self.keep_prob2)
            h_conv2 = tf.nn.dropout(tf.nn.relu(h_conv2_), self.keep_prob2)
            h_pool2 = max_pool_4x1(h_conv2)
         
            W_conv21 = weight_variable([self.conv21_filter, 1, self.dimension2, self.dimension21], 'W_conv21')
            b_conv21 = bias_variable([self.dimension21], 'b_conv21')
            #h_conv22 = tf.nn.relu(tf.nn.batch_normalization(conv2d(h_conv2, W_conv22), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001))
            h_conv21_=conv2d_1(h_pool2, W_conv21)
            h_conv21 = tf.nn.dropout(tf.nn.relu(h_conv21_), self.keep_prob2)
            
            h_pool21 = max_pool_4x1(h_conv21)
            
            W_fc1 = weight_variable([1 * self.fc1_param * self.dimension21, self.dimension4], 'W_fc1')
            b_fc1 = bias_variable([self.dimension4], 'b_fc1')
            h_pool3_flat = tf.reshape(h_pool21, [-1, 1*self.fc1_param*self.dimension21])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob3)
            
            W_fc3 = weight_variable([self.dimension4, self.dimension4], 'W_fc3')
            b_fc3 = bias_variable([self.dimension4], 'b_fc3')
            #W_fc3_t = weight_variable([dimension4, dimension4], 'W_fc3_t')
            #b_fc3_t = bias_variable_high([dimension4], 'b_fc3_t')
            #T3 = tf.sigmoid(tf.matmul(h_fc2_2_drop, W_fc3_t) + b_fc3_t)
            #C3 = tf.sub(1.0, T3)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)
            #h_high3=tf.add(tf.mul(h_fc3, T3), tf.mul(h_fc2_2_drop, C3))
            h_fc3_drop =tf.nn.dropout(h_fc3, self.keep_prob3)
            
            label_shape=self.label.shape[1]
            
            
            W_fc4 = weight_variable([self.dimension4, tf.cast(label_shape, tf.int32)], 'W_fc4')
            b_fc4 = bias_variable([label_shape], 'b_fc4')
            y_conv=tf.add(tf.matmul(h_fc3_drop, W_fc4), b_fc4)
            
            variable_dict={"W_conv1": W_conv1, "W_conv2": W_conv2,"W_conv21": W_conv21,
                            "b_conv1": b_conv1, "b_conv2": b_conv2, "b_conv21": b_conv21, 
                            #"beta2": beta1, "gamma":gamma1,
                            #"beta2": beta2, "gamma":gamma2,
                            #"beta21": beta21, "gamma":gamma21,
                            "W_fc1": W_fc1,"W_fc3": W_fc3, "W_fc4": W_fc4, 
                            "b_fc1": b_fc1, "b_fc3": b_fc3,"b_fc4": b_fc4}
            neurons_dict={"h_conv21":h_conv21, "h_conv2":h_conv2, "h_conv1":h_conv1,"h_fc1_drop": h_fc1_drop, "h_fc3_drop": h_fc3_drop}
            
            return y_conv,tf.nn.sigmoid(y_conv), variable_dict, neurons_dict,
    
    @define_scope
    def saver(self):
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
            optimizer = tf.train.AdamOptimizer(self.train_speed)
    
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
                FPR, TPR, PPV=_auc_pr(true,prob,0.5)
                FPR_list.append(FPR)
                TPR_list.append(TPR)
                PPV_list.append(PPV)
            
            return FPR_list, TPR_list, PPV_list
    
    
    
    