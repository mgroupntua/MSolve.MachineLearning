using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Tensorflow;

namespace MGroup.MachineLearning
{
	/// <summary>
	/// Interface for linear regression project.
	/// </summary>
	public interface ILinearRegression
	{

		/// <summary>
		/// Evaluates the optimal coefficients of the regression that minimize the least squares error
		/// between the given values of the dependent variables and the ones predicted.
		/// </summary>
		/// <param name="X"> A <see cref="double"/> array containing the given values of the independent variables.</param>
		/// <param name="Y"> A <see cref="double"/> array containing the corresponding values of the dependent variable.</param>
		void Train(double[] X, double[] Y);

		/// <summary>
		/// Test the accuracy of the regression model on additional data.
		/// </summary>
		/// <param name="X"> A <see cref="double"/> array containing the additional values of the independent variables.</param>
		/// <param name="Y"> A <see cref="double"/> array containing the corresponding values of the dependent variable.</param>
		void Test(double[] X, double[] Y);

		/// <summary>
		/// Utilizes the regression model to make predictions for new data.
		/// </summary>
		/// <param name="X">A <see cref="double"/> array containing new values of the independent variables,
		/// whose outcome we want to predict.</param>
		/// <returns> A <see cref="float"/> array containing the predicted values of the dependent variables, a <see cref="float"/>
		/// containing the optimal weight and a <see cref="float"/> containing the optimal bias for the model.</returns>
		(float[],float,float) Predict(double[] X);

	}
}
