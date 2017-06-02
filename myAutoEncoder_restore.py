"""
use seq2seq
two RNNs
Encoder and Decoder
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from pylab import grid
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.python.ops import variable_scope
class myAutoEncoder_restore():
    def __init__(self,max_level,hidden_dim,train_filename="data_4base_5level.csv",test_filename="None"):
        data = pd.read_csv(train_filename, header=None)

        self._class_num=8
        self._emb_dim=hidden_dim
        self._max_length=max_level
        self._batch_size=32

        self._epoch_num=200
        #self._learning_rate=0.001

        self.data_X = np.array(data)[:, 0:max_level] - 1

        #test data
        if test_filename == None:
            self.testdata = self.data_X
        else:
            testdata=pd.read_csv(test_filename,header=None)
            self.testdata=np.array(testdata)[:, 0:max_level] - 1
            self.data_X=np.concatenate((self.testdata,self.data_X),axis=0)

        self.testlength = self.getLength(self.testdata)
        self._batch_num_test= int(len(self.testdata) / self._batch_size)
        self.length = self.getLength(self.data_X)
        self.reverse_y = self.reversedata(self.data_X)
        self._batch_num = int(len(self.data_X) / self._batch_size)

        self.x = tf.placeholder(tf.int32, [None, self._max_length])  # [batch_size, num_steps] self._batch_size
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.x_one_hot = tf.one_hot(self.x, self._class_num)
        self.y = tf.placeholder(tf.int32, [None, self._max_length])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.r2rtdecoder()


    """
    Train the network
    """

    def r2rtdecoder(self):
        """
        Create a mask that we will use for the cost function

        This mask is the same shape as x and y_, and is equal to 1 for all non-PAD time
        steps (where a prediction is made), and 0 for all PAD time steps (no pred -> no loss)
        The number 30, used when creating the lower_triangle_ones matrix, is the maximum
        sequence length in our dataset
        """

        lower_triangular_ones = tf.constant(np.tril(np.ones([self._max_length, self._max_length])), dtype=tf.float32)
        seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, self.seqlen - 1), \
                               [0, 0], [self._batch_size, self._max_length])

        # RNN
        state_size=self._emb_dim
        num_classes=self._class_num

        cell = tf.contrib.rnn.BasicRNNCell(state_size)


        init_state = tf.get_variable('init_state', [1, state_size],
                                     initializer=tf.constant_initializer(0.0))
        init_state = tf.tile(init_state, [self._batch_size, 1])
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, self.x_one_hot, sequence_length=self.seqlen,initial_state=init_state)

        y_reshaped = tf.reshape(self.y,[-1])


        """
        decoder
        """
        #en_last_output = self.last_relevant(rnn_outputs, self.seqlen)
        idx = tf.range(self._batch_size) * tf.shape(rnn_outputs)[1] + (self.seqlen - 1)
        last_rnn_output = tf.gather(tf.reshape(rnn_outputs, [-1, state_size]), idx)

        with tf.variable_scope('decoder'):
            decoder_cell = tf.contrib.rnn.BasicRNNCell(self._emb_dim)
        dec_input = last_rnn_output
        dec_in_state = final_state
        dec_outputs = []
        with tf.variable_scope('multi_decoder') as scope:
            for id in range(self._max_length):
                if id > 0:
                    scope.reuse_variables()
                dec_output, dec_out_state = seq2seq_lib.rnn_decoder([dec_input], dec_in_state, decoder_cell)
                # variable_scope.get_variable_scope().reuse_variables()
                dec_input = dec_output[0]
                dec_in_state = dec_out_state
                dec_outputs += dec_output

        dec_final_output = tf.concat(dec_outputs, axis=0)

        # Softmax layer
        with tf.variable_scope('softmax'):
            #W = tf.get_variable('W', [state_size, num_classes])
            #b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
            W = tf.Variable(tf.truncated_normal([self._emb_dim, self._class_num], stddev=0.01))
            # weight = tf.Variable([self._emb_dim, self._class_num])
            b = tf.Variable(tf.constant(0.1, shape=[self._class_num, ]))

        logits = tf.matmul(dec_final_output, W) + b

        #order not the same as y
        l1 = tf.reshape(logits, [self._max_length, -1, self._class_num])
        l2 = tf.transpose(l1, [1, 0, 2])
        logits = tf.reshape(l2, [-1, self._class_num])

        preds = tf.nn.softmax(logits)

        # To calculate the number correct, we want to count padded steps as incorrect
        correct = tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), tf.int32), y_reshaped), tf.int32) * \
                  tf.cast(tf.reshape(seqlen_mask, [-1]), tf.int32)

        final_output = tf.argmax(preds,1)
        truevalue = y_reshaped

        # To calculate accuracy we want to divide by the number of non-padded time-steps,
        # rather than taking the mean
        accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(self.seqlen, tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_reshaped, logits=logits)
        loss = loss * tf.reshape(seqlen_mask, [-1])

        # To calculate average loss, we need to divide by number of non-padded time-steps,
        # rather than taking the mean
        loss = tf.reduce_sum(loss) / tf.reduce_sum(seqlen_mask)

        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        saver=tf.train.Saver()

        with tf.Session() as sess:
            #sess.run(tf.global_variables_initializer())
            saver.restore(sess, "savemodel/twornn.ckpt")
            print("Model restored.")
            learning_rate = 5 * 1e-3
            hidden_code=[]
            rnn_code=[]
            total_acc = []
            for test_batch in range(self._batch_num_test+1):
                batch_testX,batch_y, batch_testlen = self.getNextTestBatch(self._batch_size, test_batch)
                feed = {self.x: batch_testX, self.y:batch_y,self.seqlen: batch_testlen, self.learning_rate:learning_rate}
                t, f, code, lro,acc= sess.run([truevalue, final_output,final_state,last_rnn_output,accuracy], feed_dict=feed)
                code=code.reshape([-1,self._emb_dim])
                hidden_code.append(code)
                lro = lro.reshape([-1, self._emb_dim])
                rnn_code.append(lro)
                total_acc.append(acc)
                #print("Batch: "+str(test_batch))
                print("True"+str(t[0:self._max_length]))
                print("Pred"+str(f[0:self._max_length]))

            total_acc = np.mean(np.array(total_acc))
            print("Accuracy:" + str(total_acc))
            codes=np.array(hidden_code).reshape(-1,self._emb_dim)
            df=pd.DataFrame(codes[0:len(self.testdata),:])
            #file_hidden="twornn_hidden"+train_filename[4:len(train_filename)-4]+"_"+str(self._emb_dim)+".csv"
            file_hidden="code2.csv"
            df.to_csv(file_hidden,float_format='%.5f')
            #df = pd.DataFrame(np.array(rnn_code).reshape(-1, self._emb_dim))
            #df.to_csv("twornn_output_airline12.csv", float_format='%.5f')


        return

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_len = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    def getNextBacth(self, batch_size, num):
        if num<self._batch_num:
            s=num*batch_size
            e=(num+1)*batch_size
            batch_X=self.data_X[s:e,:]
            """
            reverse y, for better decoding
            """
            batch_y=self.reverse_y[s:e,:]
            batch_y[batch_y<0]=0
            batch_len=self.length[s:e]
        else:
            datasize=len(self.data_X)
            batch_X=np.zeros([batch_size,self._max_length])
            batch_y=np.zeros([batch_size,self._max_length])
            batch_len=np.ones([batch_size])
            s=num*batch_size
            if s<datasize:
                batch_X[0:datasize-s,:]=self.data_X[s:datasize,:]
                batch_y[0:datasize - s, :] = self.reverse_y[s:datasize, :]
                batch_y[batch_y < 0] = 0
                batch_len[0:datasize-s] = self.length[s:datasize]

        return batch_X, batch_y,batch_len
    def getNextTestBatch(self, batch_size, num):
        if num<self._batch_num_test:
            s=num*batch_size
            e=(num+1)*batch_size
            batch_X=self.testdata[s:e,:]
            """
            reverse y, for better decoding
            """
            batch_y=self.reverse_y[s:e,:]
            batch_y[batch_y<0]=0
            batch_len=self.testlength[s:e]
        else:
            datasize=len(self.testdata)
            batch_X=np.zeros([batch_size,self._max_length])
            batch_y=np.zeros([batch_size,self._max_length])
            batch_len=np.ones([batch_size])
            s=num*batch_size
            if s<datasize:
                batch_X[0:datasize-s,:]=self.testdata[s:datasize,:]
                batch_y[0:datasize - s, :] = self.reverse_y[s:datasize, :]
                batch_y[batch_y < 0] = 0
                batch_len[0:datasize-s] = self.testlength[s:datasize]

        return batch_X, batch_y,batch_len
    def getLength(self,data):
        # record input length
        lengths = np.zeros(len(data))
        time_step = data.shape[1]
        for i in range(len(data)):
            lengths[i]=self._max_length
            for j in range(time_step):
                if data[i][j] == -1:
                    if(lengths[i]==self._max_length):
                        lengths[i]=j

        return lengths

    def reversedata(self,data):
        time_step = data.shape[1]
        re_data = np.zeros(data.shape)-1
        for i in range(len(data)):

            temp=data[i]
            temp=temp[::-1]

            for j in range(time_step):
                if temp[j] >-1:
                    re_data[i][0:time_step-j]=temp[j:time_step]
                    break

        return re_data

    def plot(self,loss, acc):
        '''
        Plots to evaluate the convergence of standard Bayesian optimization algorithms
        '''
        plt.figure()
        epoches=np.array(range(len(loss)))
        plt.plot(epoches, loss, '-D', label="loss")

        plt.title('Loss')
        plt.xlabel('Epoches')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        grid(True)
        plt.figure()
        plt.plot(epoches, acc, '-o', label="accuracy")
        plt.title('Accuracy')
        plt.xlabel('Epoches')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)

        plt.show()

if __name__ == '__main__':

    num_base=4
    max_level=28
    hidden_dim=15
   #filename="data_"+str(num_base)+"base_"+str(max_level)+"level.csv"
    train_filename="kervec_100000.csv"
    test_filename="kervec_1000.csv"
    myAutoEncoder_restore(max_level,hidden_dim,train_filename,test_filename=test_filename)