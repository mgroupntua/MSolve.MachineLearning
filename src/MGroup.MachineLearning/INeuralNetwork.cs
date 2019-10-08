using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Tensorflow;

namespace MGroup.MachineLearning
{
	public interface INeuralNetwork
	{
		/// <summary>
		/// Set true to import the computation graph instead of building it.
		/// </summary>
		bool IsImportingGraph { get; set; }

		/// <summary>
		/// Build dataflow graph and train a neural network
		/// </summary>
		void Train(double[,] X, float[] Y);

		void Test(Session sess);

		/// <summary>
		/// Use the neural network to make predictions for new data
		/// </summary>
		float[] Predict(double[,] X);
	}
}
