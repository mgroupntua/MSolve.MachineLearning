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
		/// Build dataflow graph, train and predict
		/// </summary>
		/// <returns></returns>

		void Train(double[,] X, float[] Y);

		void Test(Session sess);

		float[] Predict(double[,] X);
	}
}
