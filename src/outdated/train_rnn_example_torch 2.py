# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 10:43:00 2020

@author: Jake Stroud
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import torch #Use CPU because it's much faster than GPU for this small setup
plt.style.use(['ggplot'])

#%% Setup numpy objects
np.random.seed(0) #Fix seed for reproducibility
N = 50; T = 1000; dt = 1; n_steps = int(T/dt); tau = 100

y_np = np.sin(np.linspace(0,2*np.pi,T))         #Target output is a sin wave

x_ic_np = np.random.normal(0,1,N)               #Rand initial condition of the dynamics
W_np = np.random.normal(0,1.5/np.sqrt(N),(N,N)) #Weight matrix
w_out_np = np.random.normal(0,1,(1,N))          #Readout weights
b_np = 0.0                                      #Bias

#Check eigs:
w, v = (np.linalg.eig(W_np))
print(np.max(np.real(w)))

#Your favourite nonlinearity:
def IO_fun(x_in):
    return np.tanh(x_in)

#%% Simulate RNN using numpy prior to any training
def sim_rnn(x_ic_f,W_f,w_out_f,b_f):
    x_f = np.zeros((n_steps,N))
    x_f[0,:] = x_ic_f
    for t in range(n_steps-1):
        x_f[t+1,:] = x_f[t,:] + (dt/tau)*(-x_f[t,:] + W_f@IO_fun(x_f[t,:]))
    
    x_out = x_f@w_out_f.T+b_f
    return x_f,x_out

#Run the RNN:
x_np,x_out_np = sim_rnn(x_ic_np,W_np,w_out_np,b_np)

#Plot some dynamics and the initial output:
plt.plot(x_np); plt.pause(0.001)
plt.plot(x_out_np,'r');plt.plot(y_np,'b')

#%% Train using pytorch (use CPU because it's much faster than GPU for this small setup)
x_ic = torch.tensor(x_ic_np, requires_grad=True)                        #Initial condition
W = torch.tensor(W_np, requires_grad=True)                              #Weight matrix
w_out = torch.tensor(w_out_np, requires_grad=True)                      #Readout weights
b = torch.tensor(0.1, dtype = torch.float64, requires_grad=True)        #Bias
y = list(torch.tensor(np.expand_dims(y_np,axis=1)))                     #Target
all_times = list(np.arange(n_steps-1))
error_fun = torch.nn.MSELoss(reduction='sum')

#Forward pass function:
# @torch.jit.script #couldn't get this line to work
def run_rnn(x):
    # cost = torch.tensor(0.0,dtype = torch.float64) #don't need it to be a tensor
    cost = 0.0
    for t in range(n_steps-1): #loop over time
        x = x + (dt/tau)*(-x + W @ torch.tanh(x))
        cost += torch.sqrt(error_fun(torch.squeeze(w_out@x+b),y[t][0]))

    return cost

optimizer = torch.optim.Adam([x_ic,W,w_out,b],lr = 0.01)
n_train_steps = 100
cost_over_epochs = np.zeros(n_train_steps) #Initialise the cost over training epochs

#Run the training:
tic = time.time() #Time it
for epoch in range(n_train_steps):
    
    cost = run_rnn(x_ic)                    #Forward pass
    cost_over_epochs[epoch] = cost.item()   #Grab cost from current epoch
    
    if (epoch+1)%10==0:
        print('Epoch: {}/{}........'.format(epoch+1, n_train_steps), end=' ')
        print("Loss: {:.4f}".format(cost_over_epochs[epoch]))
    
    #Apply grads:
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
print('time to run:', time.time() - tic)

#Plot cost over training
plt.plot(cost_over_epochs)

#Explicity convert to numpy arrays for later use
x_ic_f = np.squeeze(x_ic.detach().numpy())
W_f = W.detach().numpy()
w_out_f = w_out.detach().numpy()
b_f = b.detach().numpy()

#%% Plot outputs and dynamics after training
x_np,x_out_np = sim_rnn(x_ic_f,W_f,w_out_f,b_f)

# plt.plot(x_np);plt.pause(0.001)

#Plot output after training
plt.plot(x_out_np,'r');plt.plot(y_np,'b')















