import numpy as np
import pandas as pd
from sklearn  import datasets
from matplotlib import pyplot as plt

class LDA:
    def __init__(self,n_comp):
        self.n_comp = n_comp
        self.lda = None

    def fit(self,X,y):
        n_ftrs = X.shape[1]
        n_class = np.unique(y)

        #S_W,S_B
        mean_overall = np.mean(X)
        S_W  = np.zeros((n_ftrs,n_ftrs))
        S_B  = np.zeros((n_ftrs,n_ftrs))
        for c in n_class:
            X_c = X[y==c]
            mc = np.mean(X_c,axis=0)
            S_W+= (X_c-mc).T.dot(X_c-mc)
            n_c = X_c.shape[0]
            mdiff = (mc-mean_overall).reshape(n_ftrs,1)
            S_B += n_c*mdiff.T.dot(mdiff)
            print(S_W.shape,S_B.shape)
        A = np.linalg.inv(S_W).dot(S_B)
        v,e = np.linalg.eig(A)
        v= v.T
        idxs = np.argsort(abs(e))[::-1]
        e = e[idxs]
        v = v[idxs]
        self.lda = v[0:self.n_comp]

    def transform(self,X):
        return np.dot(X,self.lda.T)


def plotting(digits):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.show()

if __name__ == "__main__":
    digits = datasets.load_diabetes()
    print(digits.data.shape)
    # plt.gray()
    # plt.matshow(digits.images[0])
    # plotting(digits)

    lda =LDA(n_comp=2)
    X = digits.data
    y = digits.target
    lda.fit(X,y)
    X_new = lda.transform(X)

    x1 = X_new[:,0] 
    x2 =  X_new[:,1]
    print(X.shape,X_new.shape)
    plt.scatter(x1,x2,c = y,cmap=plt.cm.get_cmap('viridis',3))
    plt.colorbar()
    plt.show()
    print("LDA Finish")
  

