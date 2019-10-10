# Examples
The examples presented in this chapter demonstrate the process of creating a model that emulates a system's input-output relation.

## Linear regression example
The first example creates a linear regression model in order to find the optimal linear relationship between a set of input data and a set of output data. This example can be found in **MSolve.MachineLearning.Tests** project. 

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
    
```csharp
		// parameters
		private int trainingEpochs = 1000;
		private float learningRate = 0.01f;
```
