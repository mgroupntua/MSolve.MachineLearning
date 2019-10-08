{% include mathjax.html %}
# Models
Performing machine learning involves creating a model, which is trained on some training data and then can process additional data to make predictions. Some popular types of models used in machine learning applications include the Linear Regression model and the Artificial Neural Networks.

## Linear Regression
Linear Regression is a linear approach to modeling the relationship between a scalar response, called the dependent variable, and one or more explanatory variables, called the independent variables.

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
