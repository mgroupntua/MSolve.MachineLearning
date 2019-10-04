using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Tensorflow;

namespace MGroup.MachineLearning
{
	/// <summary>
	/// Interface of linear regression project
	/// All example should implement IExample so the entry program will find it.
	/// </summary>
	public interface ILinearRegression
	{

		/// <summary>
		/// Set true to import the computation graph instead of building it.
		/// </summary>
		bool IsImportingGraph { get; set; }

		/// <summary>
		/// Build dataflow graph, train and predict
		/// </summary>
		/// <returns></returns>

		void Train();

		void Test(Session sess);

		void Predict(Session sess);

		Graph ImportGraph();

		Graph BuildGraph();

		/// <summary>
		/// Prepare dataset
		/// </summary>
		void PrepareData(double[] item1, double[] item2);

	}
}
