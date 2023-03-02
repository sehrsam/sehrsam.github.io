import random
import numpy as np

class Perceptron:
      
    def __init__(self):

        self.w = None
        self.history = None

    def fit(self, X,y, max_runs):
             
            
            
            self.w = np.zeros((X[0].size+1))
            
            X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
            accuracy = 0
            times_run = 0
            self.history = []
            #this sets our initial w, and initializes our variables which check if the accuracy has reached 0 or the maximum run times has occurred
            while accuracy != 1 and times_run< max_runs:
                var_check = random.randint(0, X_.shape[0])-1
                acc_count= 0

                self.w += (2*(int(self.w@X_[var_check].T <0))-1)*X_[var_check]
                #we select the variable we want to check and adjust our w
                
                accuracy = self.score(X_,y)
                self.history.append(accuracy)
                times_run+=1
                #we report the accuracy



    def score(self,X,y):
        return np.mean(self.predict(X) ==y)
        #this computes how accurate our w was 

    def predict(self,X):
        return ((self.w@X.T)>=0).astype(int)
        #This returns an array of the predicted values
        
