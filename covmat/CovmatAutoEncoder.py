"""
covariance matrix as input

vector form as output
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
from pylab import grid
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq as seq2seq_lib
from tensorflow.python.ops import variable_scope


class myAutoEncoder():
    def __init__(self, max_level, hidden_dim, train_filename="kin.csv",rnnoutputfile="kernelvec.csv", test_filename=None, embedding_file=None):

        rnnoutput = pd.read_csv(rnnoutputfile, header=None)
        self.rnn_output = np.array(rnnoutput)[:,0:max_level] -1
        self.indices_in_kin = np.array(rnnoutput)[:,max_level:max_level+2]
        rnninput = pd.read_csv(train_filename, header=None)


        self._class_num = 8
        self._emb_dim = hidden_dim
        self._max_length = max_level
        self._batch_size = 128

        self._epoch_num = 300
        # TODO: deal with data_X, processing csv file
        self.rnn_input = np.array(rnninput)
        # train_ind=range(1,len(data),2)
        # self.data_X = self.data_X[train_ind,:]

        # test data
        if test_filename is None:
            self.testdata = self.rnn_input
            self.rnnoutput_test = self.rnn_output
        else:
            testdata = pd.read_csv(test_filename, header=None)
            self.testdata = np.array(testdata)
            # self.data_X = np.concatenate((self.testdata, self.data_X), axis=0)

        if embedding_file is not None:
            embedding_code=pd.read_csv(embedding_file, header=None)
            self.embedding = np.array(embedding_code)


        self.testlength = self.getLength(self.rnnoutput_test[:,0:self._max_length])
        self._batch_num_test = int(
            np.ceil(len(self.rnnoutput_test) / self._batch_size))

        # TODO : get length
        self.length = self.getLength(self.rnn_output[:,0:self._max_length])
        self.reverse_y = self.reversedata(self.rnn_output[:,0:self._max_length])
        self.reverse_testy = self.reversedata(self.rnnoutput_test[:,0:self._max_length])
        self._batch_num = int(np.ceil(len(self.rnn_output) / self._batch_size))

        # [batch_size, num_steps] self._batch_size
        if embedding_file is not None:
            self.x_embedding = tf.placeholder(tf.float32, [None, self._max_length, self.embedding.shape[1] ])
        else:
            self.x_embedding = tf.placeholder(tf.float32, [None, self._max_length, self.rnn_input.shape[1] ])
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None, self._max_length])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        self._batch_size2 = tf.placeholder(tf.int32)
        self.r2rtdecoder()

    """
    Train the network
    use a dynamic rnn, it can deal with variable lengths (requires the length for each sequence)
    """

    # TODO: more layers in RNN
    # TODO: make hidden code non-negative

    def r2rtdecoder(self):
        """
        Create a mask that we will use for the cost function
        This mask is the same shape as x and y_, and is equal to 1 for all non-PAD time
        steps (where a prediction is made), and 0 for all PAD time steps (no pred -> no loss)
        The number 30, used when creating the lower_triangle_ones matrix, is the maximum
        sequence length in our dataset
        """

        lower_triangular_ones = tf.constant(
            np.tril(np.ones([self._max_length, self._max_length])), dtype=tf.float32)
        seqlen_mask = tf.slice(tf.gather(lower_triangular_ones, self.seqlen - 1),
                               [0, 0], [self._batch_size2, self._max_length])

        # RNN
        state_size = self._emb_dim
        num_classes = self._class_num

        cell = tf.contrib.rnn.BasicRNNCell(state_size)

        init_state = tf.get_variable('init_state', [1, state_size],
                                     initializer=tf.constant_initializer(0.0))
        init_state = tf.tile(init_state, [self._batch_size2, 1])

        rnn_outputs, final_state = tf.nn.dynamic_rnn(
            cell, self.x_embedding, sequence_length=self.seqlen, initial_state=init_state)

        y_reshaped = tf.reshape(self.y, [-1])

        """
        decoder

        use the last step output of encoder as the input
        """
        # en_last_output = self.last_relevant(rnn_outputs, self.seqlen)
        idx = tf.range(self._batch_size2) * \
              tf.shape(rnn_outputs)[1] + (self.seqlen - 1)
        last_rnn_output = tf.gather(tf.reshape(
            rnn_outputs, [-1, state_size]), idx)

        with tf.variable_scope('decoder'):
            decoder_cell = tf.contrib.rnn.BasicRNNCell(self._emb_dim)
        dec_input = last_rnn_output
        dec_in_state = final_state
        dec_outputs = []
        with tf.variable_scope('multi_decoder') as scope:
            for id in range(self._max_length):
                if id > 0:
                    scope.reuse_variables()
                dec_output, dec_out_state = seq2seq_lib.rnn_decoder(
                    [dec_input], dec_in_state, decoder_cell)
                # variable_scope.get_variable_scope().reuse_variables()
                dec_input = dec_output[0]
                dec_in_state = dec_out_state
                dec_outputs += dec_output

        # dec_outputs: [batch_size, max_length, state_size]
        # [batch_size*maxlenth, state_size]
        dec_final_output = tf.concat(dec_outputs, axis=0)

        # Softmax layer
        # with tf.variable_scope('softmax'):
        # W = tf.get_variable('W', [state_size, num_classes])
        # b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
        # weight = tf.Variable([self._emb_dim, self._class_num])

        W = tf.Variable(tf.truncated_normal(
            [self._emb_dim, self._class_num], stddev=0.01))
        b = tf.Variable(tf.constant(0.1, shape=[self._class_num, ]))
        logits = tf.matmul(dec_final_output, W) + b

        # order not the same as y with tf.concat
        l1 = tf.reshape(logits, [self._max_length, -1, self._class_num])
        l2 = tf.transpose(l1, [1, 0, 2])
        logits = tf.reshape(l2, [-1, self._class_num])

        preds = tf.nn.softmax(logits)
        final_output = tf.argmax(preds, 1)
        """
        Accuracy
        """
        # To calculate the number of correctly predicted value(we want to count
        # padded steps as incorrect)
        correct = tf.cast(tf.equal(tf.cast(final_output, tf.int32), y_reshaped), tf.int32) * \
                  tf.cast(tf.reshape(seqlen_mask, [-1]), tf.int32)
        truevalue = y_reshaped
        # To calculate accuracy we want to divide by the number of non-padded time-steps,
        # rather than taking the mean
        accuracy = tf.reduce_sum(
            tf.cast(correct, tf.float32)) / tf.reduce_sum(tf.cast(self.seqlen, tf.float32))
        """
        Loss function
        """
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_reshaped, logits=logits)
        loss = loss * tf.reshape(seqlen_mask, [-1])

        # To calculate average loss, we need to divide by number of non-padded time-steps,
        # rather than taking the mean
        loss = tf.reduce_sum(loss) / tf.reduce_sum(seqlen_mask)
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        saver = tf.train.Saver()

        """
        Training
        """
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            e_loss = []
            e_acc = []
            learning_rate = 2 * 1e-3
            for epoch in range(self._epoch_num):
                total_loss = []
                total_acc = []
                for batch in range(self._batch_num):
                    batch_X, batch_y, batch_len = self.getNextBatch(
                        self._batch_size, batch)
                    batch_size_2 = batch_X.shape[0]
                    feed = {self.x_embedding: batch_X, self.y: batch_y, self.seqlen: batch_len,
                            self._batch_size2: batch_size_2, self.learning_rate: learning_rate}
                    cor, dec_out, y_re, log, acc, cost, _ = sess.run(
                        [correct, dec_outputs, y_reshaped, logits, accuracy, loss, optimizer], feed_dict=feed)
                    total_loss.append(cost)
                    total_acc.append(acc)

                total_loss = np.sum(np.array(total_loss))
                total_acc = np.mean(np.array(total_acc))
                e_loss.append(total_loss)
                e_acc.append(total_acc)
                print("Epoch" + str(epoch) + ":")
                print("Loss: " + str(total_loss) + "  " +
                      "Accuracy: " + str(total_acc))

                if total_loss < 30:
                    learning_rate = 1e-3
                if total_loss < 15:
                    learning_rate = 1e-4
                    # print("Learning rate changed.")

                if epoch == self._epoch_num - 1 or total_loss < 0.5:  # or total_acc>0.985:
                    hidden_code = []
                    rnn_code = []
                    total_acc = []
                    for test_batch in range(self._batch_num_test):
                        if test_batch == self._batch_num_test - 1:
                            a = 1
                        batch_testX, batch_y, batch_testlen = self.getNextTestBatch(
                            self._batch_size, test_batch)
                        batch_testsize_2 = batch_testX.shape[0]

                        feed = {self.x_embedding: batch_testX, self.y: batch_y, self.seqlen: batch_testlen,
                                self._batch_size2: batch_testsize_2, self.learning_rate: learning_rate}
                        last_rnno, rnno, t, f, code, acc = sess.run(
                            [last_rnn_output, rnn_outputs, truevalue, final_output, final_state, accuracy],
                            feed_dict=feed)
                        code = code.reshape([-1, self._emb_dim])
                        hidden_code.extend(code)
                        total_acc.append(acc)

                        # print("Batch: "+str(test_batch))
                        print("True" + str(t[0:self._max_length]))
                        print("Pred" + str(f[0:self._max_length]))
                    total_acc = np.mean(np.array(total_acc))
                    print("Accuracy:" + str(total_acc))
                    codes = np.array(hidden_code).reshape(-1, self._emb_dim)
                    df = pd.DataFrame(codes[0:len(self.testdata), :])
                    file_hidden = "toydata/covmat_hiddencode_split" + \
                                  str(self._emb_dim) + ".csv"
                    df.to_csv(file_hidden, float_format='%.5f')
                    break
                    # Save the variables to disk.
            # save_path = saver.save(sess, "savemodel/twornn3.ckpt")
            # print("Model saved in file: " + save_path)

            self.plot(np.array(e_loss), np.array(e_acc))

        return

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_len = int(output.get_shape()[1])
        output_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(output, [-1, output_size])
        relevant = tf.gather(flat, index)
        return relevant

    def getNextBatchEmb(self, batch_size,num):
        s = num * batch_size
        e = (num + 1) * batch_size
        batch_X = self.rnn_output[s:e,:]

        batch_X_Emb=[]
        for i in range(batch_X.shape[0]):
            X_Emb = []
            for j in range(self._max_length):
                if batch_X[i,j]>-1:
                    X_Emb.append(self.embedding[batch_X[i,j],:])
                else:
                    X_Emb.append(np.zeros(self.embedding.shape[1]))

            batch_X_Emb.append(X_Emb)

        batch_y = self.reverse_y[s:e, :]
        batch_y[batch_y < 0] = 0
        batch_len = self.length[s:e]
        batch_X_Emb = np.array(batch_X_Emb)
        return batch_X_Emb, batch_y, batch_len

    def getNextTestBatchEmb(self, batch_size, num):
        s = num * batch_size
        e = (num + 1) * batch_size
        batch_X = self.rnnoutput_test[s:e, :]

        batch_X_Emb = []
        for i in range(batch_X.shape[0]):
            X_Emb = []
            for j in range(self._max_length):
                if batch_X[i, j] > -1:
                    X_Emb.append(self.embedding[batch_X[i, j], :])
                else:
                    X_Emb.append(np.zeros(self.embedding.shape[1]))

            batch_X_Emb.append(X_Emb)

        batch_y = self.reverse_testy[s:e, :]
        batch_y[batch_y < 0] = 0
        batch_len = self.testlength[s:e]
        batch_X_Emb = np.array(batch_X_Emb)
        return batch_X_Emb, batch_y, batch_len

    def getNextBatch(self, batch_size, num):

        s = num * batch_size
        e = (num + 1) * batch_size


        batch_strings = self.indices_in_kin[s:e, :]
        batch_X = np.zeros([batch_strings.shape[0],self._max_length, self.rnn_input.shape[1]])
        for i in range(batch_strings.shape[0]):
            count = 0
            for j in range(batch_strings[i,0],batch_strings[i,1]+1,1):
                batch_X[i,count,:]=self.rnn_input[j, :]
                count += 1
        """
        reverse y, for better decoding
        """
        batch_y = self.reverse_y[s:e, :]
        batch_y[batch_y < 0] = 0
        batch_len = self.length[s:e]

        # if e exceeds the size of data, will ignore automatically, do not need to distinguish the last batch

        return batch_X, batch_y, batch_len

    def getNextTestBatch(self, batch_size, num):

        s = num * batch_size
        e = (num + 1) * batch_size
        batch_strings = self.indices_in_kin[s:e, :]
        batch_X = np.zeros([batch_strings.shape[0], self._max_length, self.rnn_input.shape[1]])
        for i in range(batch_strings.shape[0]):
            count = 0
            for j in range(batch_strings[i,0],batch_strings[i,1]+1,1):
                batch_X[i,count,:]=self.testdata[j, :]
                count += 1
        """
        reverse y, for better decoding
        """
        batch_y = self.reverse_testy[s:e, :]
        batch_y[batch_y < 0] = 0
        batch_len = self.testlength[s:e]
        batch_X = np.array(batch_X)
        return batch_X, batch_y, batch_len

    def getLength(self, data):
        # record input length
        lengths = np.zeros(len(data))
        time_step = data.shape[1]
        for i in range(len(data)):
            lengths[i] = self._max_length
            for j in range(time_step):
                if data[i][j] == -1:
                    if (lengths[i] == self._max_length):
                        lengths[i] = j

        return lengths

    def reversedata(self, data):
        time_step = data.shape[1]
        re_data = np.zeros(data.shape) - 1
        for i in range(len(data)):

            temp = data[i]
            temp = temp[::-1]

            for j in range(time_step):
                if temp[j] > -1:
                    re_data[i][0:time_step - j] = temp[j:time_step]
                    break

        return re_data

    def plot(self, loss, acc):
        '''
        Plots to evaluate the convergence of standard Bayesian optimization algorithms
        '''
        plt.figure()
        epoches = np.array(range(len(loss)))
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
    num_base = 4
    max_level = 5
    hidden_dim = 30
    # filename="data_"+str(num_base)+"base_"+str(max_level)+"level.csv"
    train_filename = "toydata/kin_split.csv"
    test_filename = "kervec_6dataset.csv"
    rnnoutput="toydata/kout.csv"
    myAutoEncoder(max_level, hidden_dim,train_filename, rnnoutputfile=rnnoutput)
