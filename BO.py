import numpy as np
import pandas as pd
import GPyOpt
import matplotlib.pyplot as plt
from pylab import savefig
from pylab import grid
import os

class bo():
    def __init__(self, iter, initInd, num_base, max_level, test_levels,hidden_dim, max_iter, dataset="airline"):

        filename2 = "twornn_hidden_"+str(num_base)+"base_"+str(max_level)+"level_"+str(hidden_dim)+".csv"
        filename1 = "onernn_hidden_" + str(num_base)+"base_"+str(max_level)+"level_"+str(hidden_dim)+".csv"
        file_nlml = "nlml_"+dataset+"_"+str(num_base)+"base_"+str(test_levels)+"levels"+".csv"
        file_org_data = "data_"+str(num_base)+"base_"+str(max_level)+"level.csv"

        data_Y = pd.read_csv(file_nlml, header=None)
        data_Y = np.array(data_Y)
        #too big nlml is not good
        valid_ind=self.data_filter(data_Y)
        self.Y = data_Y[valid_ind]

        data_twoX = pd.read_csv(filename2)
        data_twoX = np.array(data_twoX)
        data_twoX = data_twoX[valid_ind, 1:hidden_dim+1]
        
        data_oneX = pd.read_csv(filename1)
        data_oneX = np.array(data_oneX)
        #!!! data_oneX = data_oneX[0:data_Y.shape[0],1:11]
        data_oneX = data_oneX[valid_ind, 1:hidden_dim+1]

        #kernel string vectors
        data_OX = pd.read_csv(file_org_data,header=None)
        data_OX = np.array(data_OX)
        data_OX = data_OX[valid_ind,0:max_level]

        self.list_twoX = data_twoX.tolist()
        self.list_OX = data_OX.tolist()
        self.list_oneX = data_oneX.tolist()

        num_of_bo = 3
        bo_results= []

        for i in range(num_of_bo):
            if i == 0:
                self.currentList = self.list_twoX
                self.currentData = data_twoX
            elif i == 1:
                self.currentList =self.list_oneX
                self.currentData = data_oneX
            else:
                self.currentList = self.list_OX
                self.currentData = data_OX

            domain = [{'name': 'locations', 'type': 'bandit', 'domain': self.currentData}]
            initX = self.currentData[initInd]
            myBopt = GPyOpt.methods.BayesianOptimization(f=self.getNlml,  # function to optimize
                                                     domain=domain,  # box-constrains of the problem
                                                     initial_design_numdata=None,  # number data initial design
                                                     acquisition_type='EI',  # Expected Improvement
                                                     exact_feval=True,
                                                     X=initX,
                                                     maximize=False
                                                     )

            myBopt.run_optimization(max_iter,max_time=None)
            bo_results.append(myBopt.Y_best)

        #random_Y=self.random(max_iter,data_twoX.shape[0],initInd)
        #bo_results.append(random_Y)

        onehot_folder=dataset + "/" + str(test_levels) + "/" + dataset + "_onehot/"
        os.makedirs(onehot_folder,exist_ok=True)
        figurename = onehot_folder+dataset + '_'+str(hidden_dim)+'_'+str(iter) + '.png'
        min_nlml=np.min(self.Y)
        #plot and save
        self.plot_convergence(max_iter+len(initInd),min_nlml,bo_results,figurename)
        #df=pd.DataFrame(np.concatenate((self.bestY1.reshape([1,-1]),self.bestY2.reshape([1,-1]),self.bestY3.reshape([1,-1]),
        #                               random_Y.reshape([1,-1]))))

        #save results
        df = pd.DataFrame(bo_results)
        bo_folder=dataset + "/" + str(test_levels) + "/" + dataset + "_bo/"
        file_boresult=bo_folder+dataset+'_'+str(hidden_dim)+'_'+str(iter)+'.csv'
        df.to_csv(file_boresult,header=None,index=None)

    def data_filter(self, y):
        y=y.reshape([-1])
        valid_ind = np.argwhere(y < 1000)
        valid_ind = valid_ind.reshape([-1])
        # #find nan
        # bad_ind = np.argwhere(np.isnan(y)).reshape([-1])
        # #find Inf
        # bad_ind2 = np.argwhere(np.isinf(y)).reshape([-1])
        #
        # bad_ind = np.concatenate((bad_ind,bad_ind2))
        return valid_ind


    def findInd(self, x_in, list):
        x_in = x_in.ravel()
        x_in = x_in.tolist()
        ind_x = list.index(x_in)
        return ind_x

    def getNlml(self,X):
        ind_x=self.findInd(X,self.currentList)
        nlml=self.Y[ind_x]
        return nlml

    def random(self, n, dataSize,initInd):

        best_Y=np.zeros(n+len(initInd))

        X=np.random.choice(dataSize, n, replace=False)

        best=100000
        for i in range(len(initInd)):
            y=self.Y[initInd[i]]
            if y<best:
                best=y
            best_Y[i]=best

        for i in range(n):
            y=self.Y[X[i]]
            if y<best:
                best=y
            best_Y[i+len(initInd)]=best

        return best_Y

    def plot_convergence(self,n, min,bo_results , filename=None):
        '''
        Plots to evaluate the convergence of standard Bayesian optimization algorithms
        '''
        best_Y1 = np.array(bo_results[0])
        best_Y2 = np.array(bo_results[2])
        best_Y3 = np.array(bo_results[1])
        plt.figure()
        plt.plot(list(range(len(best_Y1))), best_Y1, '-D', label="twoRNN")
        plt.plot(list(range(len(best_Y2))), best_Y2, '-o', label="noRNN")
        plt.plot(list(range(len(best_Y3))), best_Y3, '-*', label="oneRNN")
        if len(bo_results)==4:
            random_Y = np.array(bo_results[3])
            plt.plot(list(range(n)), random_Y, '-+', label="Random")

        plt.plot(0, min, '*',label="minimum")
        plt.title('Value of the best selected sample')
        plt.xlabel('Iteration')
        plt.ylabel('Best y')
        #plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   #ncol=2, mode="expand", borderaxespad=0.)
        plt.legend()
        grid(True)

        if filename != None:
            savefig(filename)
        else:
            plt.show()

if __name__ == '__main__':
    dim = 15
    iter_num = 300
    data = "airline"
    #may not use all the kernel in BO, e.g. only the first 4 levels
    test_levels = 4
    bo_folder=data+"/"+str(test_levels)+"/"+data+"_bo/"
    os.makedirs(bo_folder, exist_ok=True)
    #can use same inits for different dims
    initfile="inits_"+str(80)+".csv"
    hasInit=True
    s=0
    e=15
    if hasInit==False:
        inits=[]
        for i in range(s,e):
            initInd=np.random.choice(200, 80, replace=False)
            inits.append(initInd)
            print(initInd)
            bo(i,initInd,num_base=4,max_level=5,test_levels=test_levels,hidden_dim=dim,max_iter=iter_num,dataset=data)
        df=pd.DataFrame(np.array(inits))
        df.to_csv(initfile,header=None,index=None)
    else:
        inits=pd.read_csv(initfile,header=None)
        inits=np.array(inits)
        for i in range(s,e):
            initInd=inits[i,:]
            print(initInd)
            bo(i,initInd,num_base=4,max_level=5,test_levels=test_levels,hidden_dim=dim,max_iter=iter_num,dataset=data)
