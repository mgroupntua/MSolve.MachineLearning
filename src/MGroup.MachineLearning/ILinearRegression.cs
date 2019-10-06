using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Tensorflow;

namespace MGroup.MachineLearning
{
	/// <summary>
	/// Interface of linear regression project
	/// </summary>
	public interface ILinearRegression
	{

		/// <summary>
		/// Obtain the optimal coefficients of the regression
		/// </summary>
		void Train(double[] X, double[] Y);

		void Test(Session sess);

		/// <summary>
		/// Use the computed regression coefficients to make predictions for new data
		/// </summary>
		(float[],float,float) Predict(double[] X);

	}
}
