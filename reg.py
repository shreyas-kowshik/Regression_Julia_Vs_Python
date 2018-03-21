import numpy as np 
import time

t = time.time()

reg = 0.5
alpha = 0.1

def loss(y_,y,W):
	err = y_-y
	sq_err = err*err
	loss = np.sum(sq_err)

	N = W.shape[1]
	loss = loss/(2*N)

	W[0,0] = 0
	W = W*W
	loss+=(reg*np.sum(W))

	return loss

N = 10000
D = 300
W = np.random.randn(D+1,1)
X = np.random.randn(N,D+1)
X[:,0] = 1
y = np.random.randn(N,1)
init_loss = loss(np.dot(X,W),y,W)
print("Initial Loss : ",init_loss)

for i in range(1,5000): #To reduce the loss to the same amount in julia this takes much more time
	y_ = np.dot(X,W)
	c = (1-(alpha*reg/N))
	W = W*c - (alpha/N)*(np.dot(np.transpose(X),(y_-y)))

final_loss = loss(np.dot(X,W),y,W)
print("Final Loss : ",final_loss)

# X = np.array([[1,2,3],[4,5,6],[6,5,3]])
# y = np.array([1,2,3])
# print(np.dot(X,y))
print("Loss Ratio: ",init_loss/final_loss)
t = time.time()-t 

print("Time taken : ",t)