from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.layers.recurrent import SimpleRNN, LSTM
import numpy as np
import pandas as pd
from keras import backend as K

class rnnKeras():
    def __init__(self,start,max_level, hidden_dim,filename="data_4base_5level.csv"):

        self.data_dim = max_level

        embedding=pd.read_csv("baseEmbed.csv",header=None)
        self.embedding=np.array(embedding)

        model = Sequential()
        #model.add(Masking(0,input_shape=(5,8)))
        model.add(SimpleRNN(hidden_dim,input_shape=(self.data_dim,embedding.shape[1]),return_sequences=True,activation='sigmoid'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='sgd')




        data=pd.read_csv(filename,header=None)
        data=np.array(data)
        self.a=data[start:len(data),0:self.data_dim]
        #881:5492
        #872:5495
        self.b=self.a

        ##traning with longest kernels
        (X_train, y_train), (X_test, y_test) = self.train_test_split()
        model.fit(X_train, y_train, batch_size=32, nb_epoch=10)
        loss = model.evaluate(X_test, y_test, batch_size=32)

        predict = model.predict(X_test, batch_size=32)
        predict = np.round(predict)

        for i in range(y_test.shape[0]):
            if i % 100 == 0:
                print("Pred:" + str(predict[i].T))
                print("True:" + str(y_test[i].T))
                print("------------")

        get_hidden_output = K.function([model.layers[0].input], [model.layers[0].output])
        X_all_data, record_ind = self.get_embedding(data[:, 0:self.data_dim])
        output_h = np.zeros([data.shape[0], hidden_dim])

        ##different length
        for i in range(self.data_dim):
            s = int(record_ind[i])
            e = int(record_ind[i + 1])  # don't need to -1
            output_h[s:e, :] = get_hidden_output([X_all_data[s:e, :]])[0][:, i, :]

        df = pd.DataFrame(output_h[0:len(data),:])
        df.to_csv("onernn_hidden"+filename[4:len(filename)-4]+"_"+str(hidden_dim)+".csv", float_format="%.5f")



    def _load_data(self,a,b):
        docX, docY = [], []
        for i in range(a.shape[0]):
            e_data=np.zeros([a.shape[1],self.embedding.shape[1]])
            for j in range(len(a[i])):
                if a[i][j]>0:
                    e_data[j,:]=self.embedding[int(a[i][j])-1]
                else:
                    e_data[j,:]=np.zeros(self.embedding.shape[1])
            outputdata = pd.DataFrame({"b": b[i]})

            docX.append(e_data)
            docY.append(outputdata.as_matrix())
        alsX = np.array(docX)
        alsY = np.array(docY)
        return alsX, alsY


    def get_embedding(self,a):
        record_ind = np.zeros(self.data_dim+1)

        docX=[]
        for i in range(a.shape[0]):
            e_data=np.zeros([a.shape[1],self.embedding.shape[1]])
            for j in range(len(a[i])):
                if a[i][j]>0:
                    e_data[j,:]=self.embedding[int(a[i][j])-1]
                    if record_ind[j] == 0:
                        record_ind[j]=i
                else:
                    e_data[j,:]=np.zeros(self.embedding.shape[1])
            docX.append(e_data)

        record_ind[0]=0
        record_ind[self.data_dim] = a.shape[0]
        alsX = np.array(docX)
        return alsX,record_ind


    def train_test_split(self, test_size=0.1):

        ntrn = round(self.a.shape[0] * (1 - test_size))
        X_train, y_train = self._load_data(self.a[0:ntrn],self.b[0:ntrn])
        X_test, y_test = self._load_data(self.a[ntrn:],self.b[ntrn:])

        return (X_train, y_train), (X_test, y_test)


if __name__=="__main__":
    num_base = 4
    max_level = 5
    filename = "data_" + str(num_base) + "base_" + str(max_level) + "level.csv"
    start=894
    hidden_dim=15
    rnnKeras(start,max_level,hidden_dim,filename)





