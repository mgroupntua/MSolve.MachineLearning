{% include mathjax.html %}

# Examples
The examples presented in this chapter demonstrate the process of creating a model that emulates a system's input-output relation.

## Linear regression example
This example creates a linear regression model in order to find the optimal linear relationship between a set of input data and a set of output data. Although an analytic relation exists that gives the optimal weight and bias of the regression model, in this example the problem is treated as an optimization problem for the purposes of illustration. In this context, a cost function defined as 

$$J=\frac{1}{n} \sum_{i=1}^n(pred_i-y_i)^2$$

where $pred_i=ax_i+b$, $a$ being the weight and $b$ the bias. Then, using a gradient descent algorithm the values of $a,b$ as tweaked in such a way, that the cost function gets minimized. The implementation of this example can be found in **MSolve.MachineLearning.Tests** project. 

The first code section reads a set of values **dataX** for the independent variable and the corresponding values of the dependent variable 
**dataY**.

```csharp
private static double[] dataX = {3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
			 	7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f };

private static double[] dataY = {1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
				2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f };

private static double[] testX = { 6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f };

private static double[] testY = { 1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f };
 ```

The second code section reads the parameters required from the model. These include the parameter **trainingEpochs** which refers to the maximum number of steps the gradient descent algorithm will take place in order to optimize the weight and bias of the model. The second parameter **learningRate** is used for tuning the convergence speed of the algorithm to the minimum. They are taken equal to 1000 and 0.01, respectively.
```csharp
// parameters
private int trainingEpochs = 1000;
private float learningRate = 0.01f;
```

The next step is to create the linear regression model with the specified parameters and to train it using the input data.
```csharp
LinearRegression linearRegression = new LinearRegression(trainingEpochs, learningRate);
linearRegression.Train(dataX, dataY);
```

The last step is to use this model to make predictions on new data, or extract the optimal weight and bias of the model.
```csharp
(float[] results,float optimalWeight, float optimalBias) = linearRegression.Predict(testX);
```

## Neural network example
This examples builds a neural network with a predefined architecture to solve the famous **XOR** problem. For the training of the network, a number of samples is provided which consists of a set observations from an independent set of variables along with their outcome, which is a scalar quantity. In the setting of the **XOR** problem there are two independent variables which assume discrete values of 0 and 1. If their values are equal (e.g. (0,0) or (1,1)) then the dependent variables takes the value of 0, otherwise it is equal to 1.

The first code section reads a set of values **dataX** for the independent variables and the corresponding values of the dependent variable **dataY**.

```csharp
private static double[,] dataX = { { 1, 0 }, { 1, 1 }, { 0, 0 }, { 0, 1 } };

private static float[] dataY = { 1, 0, 0, 1 };

private static double[,] testX = { { 1, 1 }, { 0, 0 }, { 0, 1 }, { 0, 1 } };

private static float[] testY = { 0, 0, 1, 1 };
 ```

The second code section reads the parameters required from the model. These include the parameter **trainingEpochs** which refers to the maximum number of steps the gradient descent algorithm will take place in order to optimize the weights and biases of the model. The second parameter **numHiddenLayers** refers to the number of hidden layers in the network architecture. They are taken equal to 2000 and 8, respectively.

```csharp
// parameters
private int trainingEpochs = 2000;
private static int numHiddenLayers = 8;
```
In certain cases, the independent variables refer to different physical quantities. In such cases, it is preferable to first normalize the data so that their values will share a common scale. A common type of normalization is the minmax feature scaling which brings all values into the range of [0 1]. To this end, the variable **normalization** is defined (even though in this example it is unnecessary, since the variables are already in the [0 1] range):

```csharp
MinMaxNormalization normalization = new MinMaxNormalization();
```

The next step is to create the neural network model with the specified parameters and to train it using the input data.
```csharp
NeuralNetwork neuralNetwork = new NeuralNetwork(numHiddenLayers, trainingEpochs, normalization);
neuralNetwork.Train(dataX, dataY);
```
The last step is to use this model to make predictions on new data.
```csharp
float[] results = neuralNetwork.Predict(testX);
```
