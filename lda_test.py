from LDA import LDA
from matplotlib import pyplot as plt
from sklearn import datasets

data = datasets.load_iris()
X = data.data
y = data.target
lda = LDA(2)

lda.fit(X,y)
X_new = lda.transform(X)

x1 = X_new[:,0] 
x2 =  X_new[:,1]
print(X.shape,X_new.shape)
plt.scatter(x1,x2,c = y,cmap=plt.cm.get_cmap('viridis',3))
plt.colorbar()
plt.show()
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()
