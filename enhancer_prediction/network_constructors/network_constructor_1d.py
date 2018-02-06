"""
    This code generates a convolutional neural network. deepshark_local_oop.py, trained_deepshark.py and deconv_deepshark_local.py import this code.
"""

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
    return FPR, TPR

class Model:
    # parameter lists
    initial_variation=0.01 #standard deviation of initial variables
    #mini batch size
    dimension1=384 #the number of the convolution filters in the 1st layer
    dimension2=384 #the number of the convolution filters in the 2nd layer
    dimension21=384
    dimension22=384
    dimension23=384
    dimension24=384
    dimension25=256
    dimension3=512 #the number of the convolution filters in the 3rd layer
    dimension4=512 #the number of the neurons in each layer of the fully-connected neural network
    conv1_filter=8
    #conv1_filter2=49
    conv2_filter=6
    conv21_filter=6
    conv22_filter=6
    conv23_filter=6
    conv24_filter=6
    conv25_filter=5
    conv3_filter=6
    
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
        self.fc1_param=int(math.ceil((math.ceil((math.ceil((math.ceil((math.ceil((math.ceil((math.ceil((#math.ceil((
            self.data_length-self.conv1_filter+1)/1.0)
                        -self.conv2_filter+1)/2.0)
                        -self.conv21_filter+1)/2.0)
                        -self.conv22_filter+1)/2.0)
                        -self.conv23_filter+1)/2.0)
                        -self.conv24_filter+1)/2.0)
                        #-self.conv25_filter+1)/1.0)
                        -self.conv3_filter+1)/2.0))
        self.prediction
        self.optimize
        self.error

        self.saver
        self.cost
        if self.output_dir is not None:
            flog=open(str(self.output_dir)+self.start_at+'.log', 'w')
            flog.write(str(__name__)+"\n"
                      +"the filer number of conv1:"+ str(self.dimension1)+"\n"
                      +"the filer size of conv1:"+ str(self.conv1_filter)+"\n"
                      +"the filer number of conv2:"+ str(self.dimension2)+"\n"
                      +"the filer size of conv2:"+ str(self.conv2_filter)+"\n"
                      +"the filer number of conv21:"+ str(self.dimension21)+"\n"
                      +"the filer size of conv21:"+ str(self.conv21_filter)+"\n"
                      +"the filer number of conv22:"+ str(self.dimension22)+"\n"
                      +"the filer size of conv22:"+ str(self.conv22_filter)+"\n"
                      +"the filer number of conv23:"+ str(self.dimension23)+"\n"
                      +"the filer size of conv23:"+ str(self.conv23_filter)+"\n"
                      +"the filer number of conv24:"+ str(self.dimension24)+"\n"
                      +"the filer size of conv24:"+ str(self.conv24_filter)+"\n"
                      #+"the filer number of conv25:"+ str(self.dimension25)+"\n"
                      #+"the filer size of conv25:"+ str(self.conv25_filter)+"\n"
                      +"the filer number of conv3:"+ str(self.dimension3)+"\n"
                      +"the filer size of conv3:"+ str(self.conv3_filter)+"\n"
                      +"the number of neurons in the fully-connected layer:"+ str(self.dimension4)+"\n"
                      +"the standard deviation of initial varialbles:"+ str(self.initial_variation)+"\n"
                      +"train speed:"+ str(self.train_speed)+"\n"
                      +"data length:" + str(self.data_length)+"\n")
            flog.close()
        

    @define_scope
    def prediction(self):

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
        def max_pool_4x1(x):
            return tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
        def max_pool_8x1(x):
            return tf.nn.max_pool(x, ksize=[1, 17, 1, 1], strides=[1, 17, 1, 1], padding='SAME')
        
        
        def batch_norm(x,beta,gamma, n_out, phase_train):
            """
            Batch normalization on convolutional maps.
            Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
            Args:
                x:           Tensor, 4D BHWD input maps
                n_out:       integer, depth of input maps
                phase_train: boolean tf.Varialbe, true indicates training phase
                scope:       string, variable scope
            Return:
                normed:      batch-normalized maps
            """
            

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            """ema = tf.train.ExponentialMovingAverage(decay=0.5)
    
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
    
            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))"""
            #normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
            normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
            return normed


        #fc1_param=int(math.ceil((math.ceil((data_length-conv1_filter+1)/4.0)-conv2_filter+1)/2.0))
        #fc1_param=int(math.ceil((math.ceil((math.ceil((math.ceil((data_length-conv1_filter+1)/1.0)-conv2_filter+1)/2.0)-conv22_filter+1)/2.0)-conv3_filter+1)/2.0))
          
        W_conv1 = weight_variable([self.conv1_filter, 4, 1, self.dimension1], 'W_conv1')
        b_conv1 = bias_variable([self.dimension1], 'b_conv1')
        #norm1=tf.nn.batch_normalization((conv2d(x_image, W_conv1) + b_conv1), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        h_conv11=conv2d_1(x_image, W_conv1)
        h_conv12=conv2d_1(x_image, tf.reverse(W_conv1, [0,1]))
        h_conv1 = tf.nn.dropout(tf.nn.relu(tf.add(h_conv11,h_conv12)), self.keep_prob)
        #h_pool1 = max_pool_2x2(h_conv1)
        #fc1_param_2=int((math.ceil((data_length-conv1_filter2+1)/8.0)))
        """beta2 = tf.Variable(tf.constant(0.0, shape=[self.dimension2]),
                                     name='beta2', trainable=True)
        gamma2 = tf.Variable(tf.constant(1.0, shape=[self.dimension2]),
                                      name='gamma2', trainable=True)  """
        W_conv2 = weight_variable([self.conv2_filter, 1, self.dimension1, self.dimension2], 'W_conv2')
        b_conv2 = bias_variable([self.dimension2], 'b_conv2')
        #h_conv2_=conv2d_1(h_conv1, W_conv2)
        #h_conv2 = tf.nn.dropout(max_pool_2x2(tf.nn.relu(batch_norm(h_conv2_,beta2,gamma2, self.dimension2,phase))), self.keep_prob2)
        h_conv2 = tf.nn.dropout(tf.nn.relu(conv2d(h_conv1, W_conv2)), self.keep_prob2)
        #h_pool2 = max_pool_2x2(h_conv2)

        """ beta21 = tf.Variable(tf.constant(0.0, shape=[self.dimension21]),
                                     name='beta21', trainable=True)
        gamma21 = tf.Variable(tf.constant(1.0, shape=[self.dimension21]),
                                      name='gamma21', trainable=True)         """ 
        W_conv21 = weight_variable([self.conv21_filter, 1, self.dimension2, self.dimension21], 'W_conv21')
        b_conv21 = bias_variable([self.dimension21], 'b_conv21')
        #h_conv22 = tf.nn.relu(tf.nn.batch_normalization(conv2d(h_conv2, W_conv22), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001))
        h_conv21_=conv2d(h_conv2, W_conv21)
        h_conv21 = tf.nn.dropout(tf.nn.relu(h_conv21_), self.keep_prob2)
        """
        beta22 = tf.Variable(tf.constant(0.0, shape=[self.dimension22]),
                                     name='beta22', trainable=True)
        gamma22 = tf.Variable(tf.constant(1.0, shape=[self.dimension22]),
                                      name='gamma22', trainable=True) """
                                      
        W_conv22 = weight_variable([self.conv22_filter, 1, self.dimension21, self.dimension22], 'W_conv22')
        b_conv22 = bias_variable([self.dimension22], 'b_conv22')
        #h_conv22 = tf.nn.dropout(tf.nn.relu(tf.nn.batch_normalization(conv2d(h_conv2, W_conv22), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)), keep_prob2)
        h_conv22_=conv2d(h_conv21, W_conv22)
        h_conv22 = tf.nn.dropout(tf.nn.relu(h_conv22_), self.keep_prob2)
        #h_conv22 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h_conv21, W_conv22),decay=0.9, center=True, scale=True,is_training=phase)), self.keep_prob2)
        #h_pool22 = max_pool_2x2(h_conv22)
        """beta23 = tf.Variable(tf.constant(0.0, shape=[self.dimension23]),
                                     name='beta23', trainable=True)
        gamma23 = tf.Variable(tf.constant(1.0, shape=[self.dimension23]),
                                      name='gamma23', trainable=True)         """
        W_conv23 = weight_variable([self.conv23_filter, 1, self.dimension22, self.dimension23], 'W_conv23')
        b_conv23 = bias_variable([self.dimension23], 'b_conv23')
        h_conv23_=conv2d(h_conv22, W_conv23)
        h_conv23 = tf.nn.dropout(tf.nn.relu(h_conv23_), self.keep_prob2)
        #h_conv23 = tf.nn.dropout(tf.nn.relu(tf.nn.batch_normalization(conv2d(h_conv22, W_conv23), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)), keep_prob2)
        #h_conv23 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h_conv22, W_conv23),decay=0.9,  center=True, scale=True,is_training=phase)), self.keep_prob2)
        #h_pool23 = max_pool_2x2(h_conv23)
        """beta24 = tf.Variable(tf.constant(0.0, shape=[self.dimension24]),
                                     name='beta24', trainable=True)
        gamma24 = tf.Variable(tf.constant(1.0, shape=[self.dimension24]),
                                      name='gamma24', trainable=True)"""   
        W_conv24 = weight_variable([self.conv24_filter, 1, self.dimension23, self.dimension24], 'W_conv24')
        #h_conv24=tf.nn.dropout(tf.nn.relu(tf.nn.batch_normalization(conv2d(h_conv23, W_conv24), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)), keep_prob3)
        #h_conv24 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h_conv23, W_conv24),decay=0.9, center=True, scale=True,is_training=phase)), self.keep_prob2)
        h_conv24_=conv2d(h_conv23, W_conv24)
        h_conv24 = tf.nn.dropout(tf.nn.relu(h_conv24_), self.keep_prob2)
        """
        W_conv25 = weight_variable([self.conv25_filter, 1, self.dimension24, self.dimension25], 'W_conv25')
        b_conv25 = bias_variable([self.dimension25], 'b_conv25')
        #h_conv24=tf.nn.dropout(tf.nn.relu(tf.nn.batch_normalization(conv2d(h_conv23, W_conv24), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)), keep_prob3)
        #h_conv25 = tf.nn.dropout(tf.nn.relu(conv2d_1(h_conv24, W_conv25)), self.keep_prob2)"""


        """beta3 = tf.Variable(tf.constant(0.0, shape=[self.dimension3]),
                                     name='beta3', trainable=True)
        gamma3 = tf.Variable(tf.constant(1.0, shape=[self.dimension3]),
                                      name='gamma3', trainable=True)      """    
        W_conv3 = weight_variable([self.conv3_filter, 1, self.dimension23, self.dimension3], 'W_conv3')
        b_conv3 = bias_variable([self.dimension3], 'b_conv3')
        h_conv3_=conv2d(h_conv24, W_conv3)
        h_conv3 = tf.nn.dropout(tf.nn.relu(h_conv3_), self.keep_prob2)
        #h_conv3=tf.nn.dropout(tf.nn.relu(tf.nn.batch_normalization(conv2d(h_conv24, W_conv3) , mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)), keep_prob3)
        #h_conv3 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(h_conv24, W_conv3), decay=0.9, center=True, scale=True,is_training=phase)), self.keep_prob2)
        #h_pool3 = max_pool_2x2(h_conv3)
        
        W_fc1 = weight_variable([1 * self.fc1_param * self.dimension3, self.dimension4], 'W_fc1')
        b_fc1 = bias_variable([self.dimension4], 'b_fc1')
        h_pool3_flat = tf.reshape(h_conv3, [-1, 1*self.fc1_param*self.dimension3])
        h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, W_fc1), b_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob3)
        
        W_fc3 = weight_variable([self.dimension4, self.dimension4], 'W_fc3')
        b_fc3 = bias_variable([self.dimension4], 'b_fc3')
        #W_fc3_t = weight_variable([dimension4, dimension4], 'W_fc3_t')
        #b_fc3_t = bias_variable_high([dimension4], 'b_fc3_t')
        #T3 = tf.sigmoid(tf.matmul(h_fc2_2_drop, W_fc3_t) + b_fc3_t)
        #C3 = tf.sub(1.0, T3)
        h_fc3 = tf.nn.relu(tf.add(tf.matmul(h_fc1_drop, W_fc3), b_fc3))
        #h_high3=tf.add(tf.mul(h_fc3, T3), tf.mul(h_fc2_2_drop, C3))
        h_fc3_drop =tf.nn.dropout(h_fc3, self.keep_prob)
        
        W_fc4 = weight_variable([self.dimension4, 1], 'W_fc4')
        b_fc4 = bias_variable([1], 'b_fc4')
        #y_conv=tf.nn.softmax(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
        y_conv=tf.add(tf.matmul(h_fc3_drop, W_fc4), b_fc4)
        variable_dict={"W_conv1": W_conv1, 
                       "W_conv2": W_conv2,
                       "W_conv21": W_conv21,
                       "W_conv22": W_conv22, 
                       "W_conv23": W_conv23,
                       "W_conv24": W_conv24,
                       "W_conv3": W_conv3, 
                       #"W_conv25": W_conv25,
                        "W_fc1": W_fc1,"W_fc3": W_fc3, "W_fc4": W_fc4, 
                        "b_fc1": b_fc1, "b_fc3": b_fc3,"b_fc4": b_fc4
                        }
        """beta2": beta2, "gamma":gamma2,
        "beta21": beta21, "gamma":gamma21,
        "beta22": beta22, "gamma":gamma22,
        "beta23": beta23, "gamma":gamma23,
        "beta23": beta24, "gamma":gamma24,
        "beta3": beta3, "gamma":gamma3"""
        neurons_dict={"h_conv3": h_conv3,  "h_conv23":h_conv23, "h_conv22":h_conv22, "h_conv24":h_conv24,
                      "h_conv21":h_conv21, "h_conv2":h_conv2, "h_conv1":h_conv1,"h_conv1":h_conv11,"h_conv1":h_conv12,"h_fc1_drop": h_fc1_drop, "h_fc3_drop": h_fc3_drop}
        
        return y_conv,tf.nn.sigmoid(y_conv), variable_dict, neurons_dict
    @define_scope
    def saver(self):
        return tf.train.Saver(var_list=self.prediction[2], max_to_keep=10)
    
    @define_scope
    def cost(self):
        #return -tf.reduce_sum(self.label * tf.log(tf.clip_by_value(self.prediction[0],1e-10,1.0))+(1-self.label)*tf.log(tf.clip_by_value(1-self.prediction[0],1e-10,1.0)))
        #return tf.reduce_mean(-tf.reduce_sum(self.label * tf.log(tf.clip_by_value(self.prediction[0],1e-10,1.0))+(1-self.label)*tf.log(tf.clip_by_value(1-self.prediction[0],1e-10,1.0)), reduction_indices=[1]))
        #return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(self.prediction[0], self.label, 9))
        #return tf.reduce_mean(-tf.reduce_sum(tf.log(tf.clip_by_value(self.label * self.prediction[0],1e-10,1.0)+tf.clip_by_value((1-self.label)*(1-self.prediction[0]),1e-10,1.0)), reduction_indices=[1]))
        return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=self.prediction[0],pos_weight=2.6))
        #return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.prediction[0]))
    @define_scope
    def optimize(self):

        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        #cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_conv*tf.log(tf.clip_by_value(2*y_conv,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))+y_*tf.log(2*tf.clip_by_value(y_,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))), reduction_indices=[1]))
        #cost = tf.reduce_sum(tf.square(y_conv*tf.log(tf.clip_by_value(2*y_conv,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))+y_*tf.log(2*tf.clip_by_value(y_,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))))
        optimizer = tf.train.AdamOptimizer(self.train_speed)

        return optimizer.minimize(self.cost)

    @define_scope
    def error(self):
        #correct_prediction = tf.equal(tf.argmax(self.prediction[1],1), tf.argmax(self.label,1))
        FPR, TPR=_auc_pr(self.label,self.prediction[1],0.5)
        return FPR, TPR
        #return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#,FPR,TPR
        #return tf.reduce_mean(tf.squared_difference(self.prediction[0], self.label))
    
    
    
    
    