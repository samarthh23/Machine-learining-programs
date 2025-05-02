import numpy as np  
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)  
y = np.array(([92], [86], [89]), dtype=float)  
x = x / np.amax(x, axis=0)  
y = y / 100  
def sigmoid(x):  
    return 1 / (1 + np.exp(-x))  
def derivatives_sigmoid(x):  
    return x * (1 - x)  
epoch = 5000  
lr = 0.1  
iln = 2  
hln = 3  
on = 1  
wh = np.random.uniform(size=(iln, hln))  
bh = np.random.uniform(size=(1, hln))  
wout = np.random.uniform(size=(hln, on))  
bout = np.random.uniform(size=(1, on))  
for i in range(epoch):  
    hinp1 = np.dot(x, wh)  
    hinp = hinp1 + bh  
    hlayer_act = sigmoid(hinp)  
      
    outinp1 = np.dot(hlayer_act, wout)  
    outinp = outinp1 + bout  
    output = sigmoid(outinp)  
      
    eo = y - output  
    outgrad = derivatives_sigmoid(output)  
    d_output = eo * outgrad  
      
    eh = d_output.dot(wout.T)  
    hiddengrad = derivatives_sigmoid(hlayer_act)  
    d_hiddenlayer = eh * hiddengrad  
      
    wout += hlayer_act.T.dot(d_output) * lr  
    wh += x.T.dot(d_hiddenlayer) * lr  
print("Input:\n",str(x))  
print("Actual output:\n",str(y))  
print("Predicted output:\n",output)  