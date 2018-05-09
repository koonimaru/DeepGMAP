import functools
import tensorflow as tf
from auc_calc import auc_pr as ac

#the code design came from https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2

def doublewrap(function):
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


class template_model(object):


    def __init__(self, label, prediction, max_to_keep, train_speed, GPUID):
        #self.label=label
        #self.prediction=prediction
        self.max_to_keep=max_to_keep
        self.train_speed=train_speed
        self.optimize
        self.error(label, prediction)
        self.saver
        self.cost(prediction)
        self.GPUID=GPUID

    @define_scope
    def saver(self):
        return tf.train.Saver(max_to_keep=self.max_to_keep)
    
    @define_scope
    def cost(self, prediction):
        with tf.device('/device:GPU:'+self.GPUID):
            nll=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self.label, logits=prediction[0],pos_weight=1.0))
            l2_norm=tf.reduce_sum(prediction[4])
            l1_norm=tf.reduce_sum(tf.abs(prediction[1]))
            return tf.add_n([nll,tf.multiply((5*10**-7), l2_norm),tf.multiply((1*10**-8),l1_norm)])

    @define_scope
    def optimize(self):
        with tf.device('/device:GPU:'+self.GPUID):
            optimizer = tf.train.AdamOptimizer(self.train_speed)
            return optimizer.minimize(self.cost)

    @define_scope
    def error(self,label, prediction):
        with tf.device('/device:GPU:'+self.GPUID):
            class_n=label.shape[1]
            FPR_list=[]
            TPR_list=[]
            PPV_list=[]
            for i in range(class_n):
                
                true=label[:,i]
                prob=prediction[1][:,i]
                FPR, TPR, PPV=ac(true,prob,0.5)
                FPR_list.append(FPR)
                TPR_list.append(TPR)
                PPV_list.append(PPV)
            
            return FPR_list, TPR_list, PPV_list
