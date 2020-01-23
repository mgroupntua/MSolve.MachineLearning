using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Accord.Math.Decompositions;

namespace MGroup.MachineLearning.UtilityFunctions
{
	public class EigenDecomposition
	{
		public static (double[], double[,]) FindEigenValuesAndEigenvectorsSymmetricOnly(double[,] dataSet, int numberOfEigs)
		{
			bool sort = true;
			bool inPlace = true;
			var evd = new EigenvalueDecomposition(dataSet, inPlace, sort);
			double[] lambdaAll = evd.RealEigenvalues;
			double[] lambda = lambdaAll.Skip(0).Take(numberOfEigs).ToArray();
			double[,] EigenvectorsAll = evd.Eigenvectors;

			return (lambda, EigenvectorsAll);

		}

	}
}
