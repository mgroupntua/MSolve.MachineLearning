using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Tensorflow;

namespace MGroup.MachineLearning
{
	/// <summary>
	/// Interface for Neural Network project.
	/// </summary>
	public interface INeuralNetwork
	{
		/// <summary>
		/// Evaluates the weight coefficients and biases of the neural network that minimize the mean-square error
		/// between the given values of the dependent variables and the ones predicted.
		/// </summary>
		/// <param name="X">A <see cref="double"/> array containing the given values of the independent variables.</param>
		/// <param name="Y">A <see cref="double"/> array containing the corresponding values of the dependent variable.</param>
		void Train(double[,] X, float[] Y);

		/// <summary>
		/// Tests the accuracy of the neural network on additional data.
		/// </summary>
		/// <param name="X">A <see cref="double"/> array containing the additional values of the independent variables.</param>
		/// <param name="Y">A <see cref="double"/> array containing the corresponding values of the dependent variable.</param>
		void Test(double[,] X, double[] Y);

		/// <summary>
		/// Utilizes the neural network to make predictions for new data.
		/// </summary>
		/// <param name="X">A <see cref="double"/> array containing the new values of the independent variables, whose outcome we want to predict.</param>
		/// <returns> A <see cref="float"/> array containing the predicted values of the dependent variables.</returns>
		float[] Predict(double[,] X);
	}
}
