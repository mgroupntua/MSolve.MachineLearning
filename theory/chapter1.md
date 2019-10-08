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

- The sigmoid function: $S(z)=\frac{1}{1+e^{-z}}$
