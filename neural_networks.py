import numpy as np
from sklearn import datasets
import tqdm
class neural_network:
    
    def __init__(self,nn_layers,epochs,l_rate):
        self.epochs = epochs
        self.lr = l_rate
        self.model = {} # nodes ,layers and weights informations
        self.nn_state = {} # activation for each layer informations
        self.layers = nn_layers # array of #nodes in input,hidden and output layer
        
    def sigmoid(self,x,derivative = False):
        if derivative:
            return np.exp(-x) / ((np.exp(-x) + 1) ** 2)
        else:
            return 1 / (1 + np.exp(-x))
        
    def nn_init(self):

        layers = self.layers
        
        #defining layer architecture for the model
        for l in range(len(layers)):
            if l == 0:
                self.model['inp_layer'] = layers[l]
            elif l == len(layers)-1:
                self.model['out_layer'] = layers[l]
            else:
                key = 'hid_layer' + str(l)
                self.model[key] = layers[l]

        # initialising weights to the model
        for i in range(1,len(layers)):
                key = 'w'+str(i)
                row = layers[i]
                colm = layers[i-1]
                self.model[key]= np.random.randn(row,colm) * np.sqrt(1. / row)


                

    def feed_forward(self,x,y): #compute activations
    
        layers = self.layers
        
        # computing activation state for the nueral network
        self.nn_state['a0'] = x
        
        for i in range(1,len(layers)):

                key1 = 'z'+ str(i)
                val = np.dot(self.model['w'+ str(i)], self.nn_state['a'+ str(i-1)])
                key2 = 'a'+ str(i)
                self.nn_state[key1] = val
                self.nn_state[key2] = self.sigmoid(val)

        self.nn_state['y'] = y
       

    def backward_propogation(self, y): #compute gradients and deltas
    
        state = self.nn_state
        state['y'] = y
        model = self.model
        layer = self.layers

        for i in range(len(layer)-1,0,-1):
            delta  = 'delta_a'+ str(i)
            grad = 'grad_w' +   str(i)
            actv  =  'a'+       str(i)
            z    = 'z' +        str(i)
            actv_prev = 'a' +   str(i-1)

            if i == len(layer)-1:
                state[delta] = 2*(state[actv] - state['y'])
                state[grad]  = np.multiply(self.sigmoid(state[z],derivative=True),state[delta]).dot(state[actv_prev].T)

            else:
                delta_next = 'delta_a'+ str(i+1)
                wt_next = 'w' + str(i+1)
                z_next = 'z'+ str(i+1)
                
                state[delta] = (model[wt_next].T).dot(np.multiply(self.sigmoid(state[z_next],derivative=True),state[delta_next]))

                state[grad] = np.multiply(self.sigmoid(state[z],derivative=True),state[delta]).dot(state[actv_prev].T)


        self.nn_state = state

    def update_weights(self):

        layers = self.layers

        for i in range(1,len(layers)):
            wt_key = 'w'+str(i)
            grad_key = 'grad_w'+str(i)
 
            self.model[wt_key]-= self.lr*self.nn_state[grad_key]





    def cost(self,pred,out):
        loss = np.sum((pred-out)**2)
        return loss

    def train(self,X_train,Y_train):
        # train
        print('################### training ####################')
        ## initialise weights
        self.nn_init()

        for e in range(self.epochs):
            print('epoch:', e)
            
            samples = X_train.shape[0]
            loss = 0
            hit_count = 0
            for i in range(samples):
                
                x_train = X_train[i].reshape(-1,1)
                y_train = Y_train[i].reshape(-1,1)
                
                
                self.feed_forward(x_train,y_train)
               
                self.backward_propogation(y_train)
                
                
                
                 #update weights
                self.update_weights()
               
                

                # add partial cost
                final_layer = 'a'+ str(len(self.layers)-1)
                loss += self.cost(self.nn_state[final_layer], y_train)
                
                if np.argmax(self.nn_state[final_layer]) == np.argmax(y_train):
                    # successful detection
                    hit_count += 1
             
        # performance evaluation
            loss = loss / samples
            accuracy = hit_count / samples
            print('Train loss:', loss, 'Train accuracy:', accuracy)
       


    def predict(self,X_test,Y_test):
        # test
        print('################### testing ####################')
        samples = X_test.shape[0]
        loss = 0
        hit_count = 0
        for i in range(samples):
            x_test = X_test[i].reshape(-1,-1)
            y_test = Y_test[i].reshape(-1,-1)
            print(x_test.shape,y_test.shape)
            self.feed_forward(x_test,y_test)
            final_layer = 'a'+ str(len(self.layers)-1)
            loss += self.cost(self.nn_state[final_layer], y_test)
            if np.argmax(self.nn_state[final_layer]) == np.argmax(y_test):
                hit_count += 1
        # evaluate performance
        loss = loss / samples
        accuracy = hit_count / samples
        print('Test loss:', loss, 'Test accuracy:', accuracy)
            
if __name__ == "__main__":
    digits = datasets.load_digits()
#     print(digits.data.shape)
    X = digits.data
    y = digits.target
    b = np.zeros((y.size,y.max()+1))
    b[np.arange(y.size), y] = 1
    lr = 0.005
    epoch = 10
    print(X.shape,b.shape)
    layers = [X.shape[1],16,12,b.shape[1]]
    nnmodel = neural_network(layers,epoch,lr)
    nnmodel.train(X[:1000],b[:1000])
    nnmodel.predict(X[1000:],b[1000:])
