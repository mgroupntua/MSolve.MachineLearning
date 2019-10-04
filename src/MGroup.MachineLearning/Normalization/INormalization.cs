using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning.Normalization
{
	public interface INormalization
	{
		(double[,], float[]) Normalize(double[,] rawX, float[] rawY);
	}
}
