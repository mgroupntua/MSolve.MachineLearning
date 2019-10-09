{% include mathjax.html %}
# Models
Performing machine learning involves creating a model, which is trained on some training data and then can process additional data to make predictions. Some popular types of models used in machine learning applications include the Linear Regression model and the Artificial Neural Networks.

## Linear Regression
Linear regression is a linear approach to modeling the relationship between a scalar response, called the dependent variable, and one or more explanatory variables, called the independent variables.

Given a data set $\{y_i,x_{i1},x_{i2},...,x_{ip}\} _ {i=1}^n$ of $n$ observations, a linear regression model assumes that the relationship between the dependent variable $y$ and the $p$-vector of regressors $\boldsymbol{x}$ is linear. The relationship takes the form

$y_i=\beta_0+\beta_1x_{i1}+...+\beta_px_{ip}+\epsilon_i=\boldsymbol{x} _ i^T \boldsymbol{\beta}+\epsilon_i$,    $i=1,...,n$

where $\epsilon$ is an error variable added in the summation. Also, $\beta_0$ is called the bias and $\beta_1,...,\beta_b$ the weights of the regression.

The above system of $n$ equation can be written in matrix form as 

$\boldsymbol{y}=\boldsymbol{X}\boldsymbol{\beta}+\boldsymbol{\epsilon}$

where 
$\boldsymbol{y}=\begin{pmatrix} y_1 \\ \vdots \\ y_n\end{pmatrix}$
$\boldsymbol{X}=\begin{pmatrix} 1 & x_{11} & \cdots & x_{ip} \\ 
                                \vdots & \ddots & \vdots \\
                                1 & \cdots & x_{np} \end{pmatrix}$
$\boldsymbol{\beta}=\begin{pmatrix} \beta_0 \\ \vdots \\ \beta_p\end{pmatrix}$
$\boldsymbol{\epsilon}=\begin{pmatrix} \epsilon_1 \\ \vdots \\ \epsilon_n\end{pmatrix}$

Using the least-squares method, the values of $\boldsymbol{\beta}$ are chosen such that the sum of the square of the errors is minimized, that is, 

$\hat{\boldsymbol{\beta}}=argmin_{\boldsymbol{\beta}} \sum_{i=1}^n \epsilon_i^2=argmin_{\boldsymbol{\beta}} \sum_{i=1}^n (\boldsymbol{\beta} \dot \boldsymbol{x}_ i )^2$.

Finding the values of $\boldsymbol{\beta}$ that minimize the above quantity constitutes a problem of convex optimization and the optimal values can be obtained via a gradient descent algorithm. Also, for this type of problems an analytical solution is given by the expression

$\boldsymbol{\beta}=(\boldsymbol{X}^T\boldsymbol{X})^{-1}\boldsymbol{X}\boldsymbol{y}$

## Artificial Neural Networks
Artificial neural networks (ANNs) are computing systems inspired by biological neural networks that constitute the brain. An ANN is based on a collection of connected units and cells called artificial neurons, which, in a sense, model the neurons in a biological brain. Each connection can transmit a signal to other neurons, similarly to the synapses in a the brain.

In ANN implementations the simplest architecture is that of the Perceptron, where artificial neurons -known as linear threshold units (LTU) in this context- consist of one or more numerical inputs and one numerical output. The LTU computes a weighted sum of its inputs, that is, $z=w_1x_1+w_2+x_2+...+w_nx_n=\boldsymbol{w}^T\boldsymbol{x}$, then applies a step function to that sum and outputs the result:

$h_{\boldsymbol{w}}(\boldsymbol{x})=step(z)=step(\boldsymbol{w}^T\boldsymbol{x})$

Some common step functions include:

- The Heaviside function: $H(z)= \begin{cases} 0 \ \ \ \text{if} \ z<0 \\ 1 \ \ \ \text{if} \ z>0 \end{cases}$

- The sign function: $sgn(z)=\begin{cases} -1 \ \ \ \text{if} \ z<0 \\ 0 \ \ \ \text{if} \ z=0 \\ 1 \ \ \ \text{if} \ z>0\end{cases}$

A Perceptron is simply composed of a single layer of LTUs, with each neuron connected to all the inputs. Training of the network refers to the process of ajusting the connection weights $w_{i,j}$ between neurons $i$ and $j$ so that a loss function is minimized. More specifically, the weights are updated as

$w_{i,j}^{(next step)}=w_{i,j}}+\eta(\hat{y}_ j-y_ j)x_i$

where 
- $x_i$ is the $i^{th}$ input value of the current training instance
- $\hat{y}_ j$ is the output of the $j^{th}$ output neuron for the current training instance
- $\eta$ is the learning rate

AN improved version of Perceptrons, capable of achieving more accurate results in complex problems, is that of the Multi-Layer Perceptron (MLP), which is the result of stacking multiple Perceptrons. An MLP is composed of one input layer, one or more layers of LTUs, called hidden layers, and one final layer of LTUs called the output layer. Every layer except the output layer includes a bias neuron and is fully connected to the next layer. When an ANN has two or more hidden layers, it is called a deep neural network (DNN).

The most common training algorithm for DNNs is the backpropagation training algorithm. The idea for this algorithm is the following.
For each training instance, the algorithm feeds it to the network and computes the output of every neuron in each consecutive layer. Then, it measures a loss function that represents the network's output error, which is the difference between the desired output and the actual output of the network, and it also computes how much each neuron in the last hidden layer contributed to each output neuron's error. It then proceeds to measure how much of these error contributions came from each neuron in the previous hidden layer, and this process is repeated until the algorithm reaches the input layer. This process measures the error gradient across all the connection weights in the network by propagating the error gradient backward in the network. Then, it finally tweaks the connection weights to reduce the error.

Algorithmically, the backpropagation algorithm consists of the following steps. Let's assume, for simplicity, one output neuron.

1) We define an error function $L(t,y)=E$, e.g., $E=(t-y)^2$, where $E$ is the loss function, $t$ is the target output of the training sample and $y$ is the actual output of the output neuron. For each neuron $j$, its output $o_j$ is defined as

$o_j=\phi(net_j)=\phi(\sum_{k=1}^n w_{kj}o_k)$

where $\phi$ is the activation function, $w_{kj}$ denotes the weight between neuron $k$ of the previous layer and neuron $j$ of the current layer, and $net_j$ is the input to neuron, which is equal to th weighted sum of the outputs $o_k$ of previous neurons. 

Some popular activation functions for the backpropagation algorithm are
- The sigmoid function: $S(z)=\frac{1}{1+e^{-z}}$
- The hyperbolic tangent function: $tanh(z)=2S(2z)-1$
- The ReLU function: ReLU(z)=max(0,z)

2) The weights $w_{ij}$ in the network are adjusted, in order to minimize the error $E$. This is achieved by performing a gradient descent algorithm that calculates the partial derivate of the error with respect to each weight $w_{ij}$ as follows

$\frac{\partial E}{\partial w_}{ij}=\frac{\partial E}{\partial o_j}\frac{\partial o_j}{\partial w_{ij}}=\frac{\partial E}{\partial o_j}\frac{\partial o_j}{\partial net_j}\frac{\partial net_j}{\partial w_{ij}}$

Notice that,

$\frac{\partial net_j}{\partial w_{ij}}=\frac{\partial}{\partial w_{ij}}(\sum_{k=1}^n w_{kj}o_k)=o_i$

$\frac{\partial o_j}{\partial net_j}=\frac{\partial \phi(net)_ j}{\partial net_j}$

and 

$\frac{\partial E}{\partial o_j}=\frac{\partial E}{\partial y}$ if the neuron is in the output layer

or, it is given by the recursive formula

$\frac{\partial E}{\partial o_j}=\sum_{m \in M}(\frac{\partial E}{\partial o_m}\frac{\partial 0_m}{\partial net_m}w_{jm})

where $M$ is the subset of the neurons, from which neuron $j$ receives input. Putting all these together we end up with the following expression for the partial derivatives

$\frac{\partial E}{\partial w_{ij}}=o_i\delta_j $

with

$\delta_j=\begin{cases} \frac{\partial L(o_j,t)}{\partial o_j}\frac{d \phi(net_j)}{d net_j} \ \ \text{ij } j \ \text{is an output neuron}\\ (\sum_{m \in M} w_{jm}\delta_m) \frac{d \phi(net_j)}{d net_j} \ \ \text{ij } j \ \text{is an inner neuron} $

To update the weight $w_{ij}$ using the gradient descent algorithm, a learning rate $c>0$ must be chosen, which refers to the step size of the algorithm. Then, the updated weights are given by

$w_{ij}=w_{ij}+\Delta w_{ij}=w_{ij}-c\frac{\partial E}{\partial w_{ij}}=-c o_i \delta_j$








# References
[1] Machine Learning with Scikit-Learn & TensorFlow, A. Geron, O' Reilly, 2017
