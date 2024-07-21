import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sig_derivative(x):
    return x*(1-x)

and_not_X=np.array([[0,0],[0,1],[1,0],[1,1]])
and_not_y=np.array([[0],[0],[1],[0]])

xor_X=np.array([[0,0],[0,1],[1,0],[1,1]])
xor_y=np.array([[0],[1],[1],[0]])

class MLP:
    def __init__(self,input_size,hidden_size,output_size):
        self.weights_input_hidden=np.random.rand(input_size,hidden_size)
        self.weights_hidden_output=np.random.rand(hidden_size,output_size)
        self.bias_hidden=np.random.rand(1,hidden_size)
        self.bias_output=np.random.rand(1,output_size)

    def forward(self,X):
        self.hidden=sigmoid(np.dot(X,self.weights_input_hidden)+self.bias_hidden)
        self.output=sigmoid(np.dot(self.hidden,self.weights_hidden_output)+self.bias_output)
        return self.output

    def backward(self,X,y,output):
        output_error=y-output
        output_delta=output_error*sig_derivative(output)
        hidden_error=output_delta.dot(self.weights_hidden_output.T)
        hidden_delta=hidden_error*sig_derivative(self.hidden)
        self.weights_hidden_output+=self.hidden.T.dot(output_delta)
        self.weights_input_hidden+=X.T.dot(hidden_delta)
        self.bias_hidden+=np.sum(hidden_delta,axis=0,keepdims=True)
        self.bias_output+=np.sum(output_delta,axis=0,keepdims=True)

    def train(self,X,y,epochs):
        for _ in range(epochs):
            output=self.forward(X)
            self.backward(X,y,output)

    def predict(self,X):
        return(self.forward(X)>0.5).astype(int)

and_not_mlp=MLP(input_size=2,hidden_size=4,output_size=1)
and_not_mlp.train(and_not_X,and_not_y,epochs=5000)

xor_mlp=MLP(input_size=2,hidden_size=4,output_size=1)
xor_mlp.train(xor_X,xor_y,epochs=5000)

print('and_not:')
print(and_not_mlp.predict(and_not_X))

print('xor:')
print(xor_mlp.predict(xor_X))
