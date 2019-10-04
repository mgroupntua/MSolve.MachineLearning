namespace MGroup.MachineLearning.Tests
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.Drawing;
	using System.Linq;
	using System.Reflection;

	using MGroup.MachineLearning;
	using NumSharp;
	using Tensorflow;
	using Xunit;

	using static Tensorflow.Binding;

	public class LinearRegressionExample
	{
		private float optimizedWeight;
		private float optimizedBias;

		// data input
		readonly double[] inputX = {3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
			 7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f };

		readonly double[] inputY = {1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
						 2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f };

		readonly double[] testX = { 6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f };

		readonly double[] testY = { 1.84f, 2.273f, 3.2f, 2.831f, 2.92f, 3.24f, 1.35f, 1.03f };

		// parameters
		private int trainingEpochs = 1000;
		private float learningRate = 0.01f;
		private int displayStep = 50;
		NumPyRandom rng = np.random;

		LinearRegression linearRegression = new LinearRegression();

		private void AssignParameters()
		{
			linearRegression.LearningRate = learningRate;
			linearRegression.DisplayStep = displayStep;
			linearRegression.TrainingEpochs = trainingEpochs;
			linearRegression.InputX = inputX;
			linearRegression.InputY = inputY;
		}

		[Fact]
		private void TestLinearRegression()
		{
			AssignParameters();
			linearRegression.Train();
			optimizedWeight = linearRegression.Weight;
			optimizedBias = linearRegression.Bias;
			//analytical solution coefficientVector=(X'*X)*X'*Y where X is the inputX with ones in the first column and Y is inputY: bias=0.7988 and weight=0.2516

			Assert.Equal(0.309068531, optimizedWeight, 6);
			Assert.Equal(0.373608261, optimizedBias, 6);
		}
	}
}
