using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning.Normalization
{
	/// <summary>
	/// Normalize the data using the
	/// MinMax normalization so that 
	/// their values lie in
	/// the [0,1] domain.
	/// </summary>
	public class MinMaxNormalization : INormalization
	{
		public (double[,], float[]) Normalize(double[,] rawX, float[] rawY)
		{
			throw new NotImplementedException();
		}
	}
}
