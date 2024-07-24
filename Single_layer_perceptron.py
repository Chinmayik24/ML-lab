import numpy as np

def step(x):
    return np.where(x>=0,1,0)

and_X=np.array([[0,0],[0,1],[1,0],[1,1]])
and_y=np.array([[0],[0],[0],[1]])

or_X=np.array([[0,0],[0,1],[1,0],[1,1]])
or_y=np.array([[0],[1],[1],[1]])

class perceptron:
    def __init__(self,input_size,learning_rate=0.1,epochs=1000):
        self.weights=np.zeros((input_size,1))
        self.bias=0
        self.learning_rate=learning_rate
        self.epochs=epochs

    def train(self,X,y):
        for _ in range(self.epochs):
            for inputs,labels in zip(X,y):
                inputs=inputs.reshape(-1,1)
                learning_output=np.dot(inputs.T,self.weights)+self.bias
                prediction=step(learning_output)
                error=labels-prediction
                self.weights+=error*self.learning_rate*inputs
                self.bias+=error*self.learning_rate

    def predict(self,X):
        linear_output=np.dot(X,self.weights)+self.bias
        return step(linear_output)

and_percep=perceptron(input_size=2)
and_percep.train(and_X,and_y)

or_percep=perceptron(input_size=2)
or_percep.train(or_X,or_y)

print('and: ',and_percep.predict(and_X))
print('or: ',or_percep.predict(or_X))
