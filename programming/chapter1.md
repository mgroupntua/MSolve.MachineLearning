# Examples
The examples presented in this chapter demonstrate the process of creating a model that emulates a system's input-output relation.

## Linear regression example
The first example creates a linear regression model in order to find the optimal linear relationship between a set of input data and a set of output data. Although an analytic relation exists that gives the optimal weight and bias of the regression model, in this example the problem is treated as an optimization problem for illustrative purposes. In this context, a cost function defined as 

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

