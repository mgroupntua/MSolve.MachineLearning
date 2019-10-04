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
	using Normalization;

	using static Tensorflow.Binding;
	public static class NeuralNetworkExample
	{
		// data input
		private static double[,] dataX = { { 1, 0 }, { 1, 1 }, { 0, 0 }, { 0, 1 } };

		private static float[] dataY = { 1, 0, 0, 1 };

		private static double[,] testX = { { 1, 1 }, { 0, 0 }, { 0, 1 }, { 0, 1 } };

		private static float[] testY = { 0, 0, 1, 1 };
		// parameters
		private static int trainingEpochs = 2000;
		private static int numHiddenLayers = 8;

		[Fact]
		private static void TestNeuralNetwork()
		{

			MinMaxNormalization normalization = new MinMaxNormalization();

			NeuralNetwork neuralNetwork = new NeuralNetwork(numHiddenLayers, trainingEpochs, normalization);
			neuralNetwork.Train(dataX, dataY);
			float[] results = neuralNetwork.Predict(testX);

			float[] errorNorm = new float[testY.GetLength(0)];
			for (int i = 0; i < testY.GetLength(0); i++)
			{
				errorNorm[i] = (float)Math.Pow(results[i] - testY[i], 2);
			}
			double totalError = errorNorm.Sum();
			Assert.True(totalError < 0.01);
		}
	}
}
