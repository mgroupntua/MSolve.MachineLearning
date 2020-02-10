using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning.Normalization
{
	public interface INormalization
	{
		(double[,], double[], double[]) Normalize(double[,] rawX);   
	}
}
