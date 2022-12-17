import numpy as np 
from sklearn  import datasets
from matplotlib import pyplot as plt

class MLP:
    def __init__(self,X,y,layers,nodes,epoch,l_rate) :
        self.X = X
        self.y = y
        self.output = np.unique(y)
        self.input =  X.shape[1]
        self.layers = layers
        self.h1 = nodes
        self.h2 = nodes

        self.epoch = epoch
        self.lr = l_rate
    
    def weights_biases(self):
        w1 = np.ones((self.h1,self.input))
        b1 = np.ones((self.h1,1))
        w2 = np.ones((self.h2,self.h1))
        b2 = np.ones((self.h2,1))
        w3 = np.ones((self.output,self.h2))
        b3 = np.ones((self.h3,1))
        para = {"w1":w1,"w2":w2,"w3":w3,"b1":b1,"b2":b2,"b3":b3}
        return para

    
    def sigmoid(self,V):
        V_new = 1/(1+np.exp(-V))
        return V_new

    def sigmoid_dash(self,x):
        sigx = self.sigmoid(x)
        sigdashx = sigx*(1-sigx)
        return sigdashx


    def feedforward(self,X,para):
        w1,w2,w3 = para["w1"],para["w3"],para["w3"]
        b1,b2,b3 = para["b1"],para["b2"],para["b3"]
        z1 = X.dot(w1.T) + b1
        a1 = self.sigmoid(z1)
        z2 = a1.dot(w2.T) + b2
        a2 = self.sigmoid(z2)
        z3 = a2.dot(w3.T)+b3
        a3 = self.sigmoid(z3)

        activations= {"z1":z1,"z2":z2,"z3":w3,"a1":a1,"a2":a2,"a3":a3}
        return activations

        
    def backpropagation_update_weights(self,activations,para,inp,out):
        a1,a2,a3 = activations["a1"],activations["a2"],activations["a3"]
        z1,z2,z3 = activations["z1"],activations["z2"],activations["z3"]
        w1,w2,w3  = para["w1"],para["w2"],para["w3"]
        b1,b2,b3  = para["b1"],para["b2"].para["b3"]

        delta3 = 2*(a3 - out)
        grad_w3 = np.multiply(self.sigmoid_dash(z3),delta3) .dot(a2.T)

        delta2 = (w3.T).dot(np.multiply(self.sigmoid_dash(z3),delta3))
        grad_w2 = np.multiply(self.sigmoid_dash(z2),delta2).dot(a1.T)

        delta1 = (w2.T).dot(np.multiply(self.sigmoid_dash(z2),delta2))
        grad_w1 = np.multiply(self.sigmoid_dash(z1),delta1).dot(inp.T)

        w1 = w1 -grad_w1*self.lr
        w2 = w2 -grad_w2*self.lr
        w3 = w3 -grad_w3*self.lr

        para["w1"] = w1
        para["w2"] = w2
        para["w3"] = w3

        return para



        

    def cost(self,pred,out):
        loss = np.sum((pred-out)**2)
        return loss


    def train(self):

        para = self.weights_biases()
        w1,w2,w3  = para["w1"],para["w2"],para["w3"]
        b1,b2,b3  = para["b1"],para["b2"].para["b3"]
        loss = []
        for e in range(self.epochs):
            
            for idx in range(len(self.X)):
                x_train = self.X[idx]
                y_train = self.y[idx]


                activations = self.feedforward(x_train,para)
                current_loss = self.cost(para["a3"],y_train)
                loss.append(current_loss)
                para        = self.backpropagation_update_weights(activations,para)
                print(f"epoch = {e} and loss = {current_loss}")


        plt.plot(loss)
        plt.grid()
        plt.show()



if __name__ == "__main__":
    digits = datasets.load_diabetes()
    print(digits.data.shape)
    X = digits.data
    y = digits.target

    mlp = MLP(X,y,2,16,4,0.0003)
    mlp.train()

        

               
                        


        

    

 