using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Accord.Math.Decompositions;

namespace MGroup.MachineLearning.UtilityFunctions
{
	public class EigenDecomposition
	{
		public static (double[], double[,]) FindEigenValuesAndEigenvectorsSymmetricOnly(double[,] dataSet, int numberOfEigs, bool inPlace, bool sort, bool scaled)
		{
			double tempSum;
			double maxEigVal;
			var evd = new EigenvalueDecomposition(dataSet, inPlace, sort);
			double[] lambdaAll = evd.RealEigenvalues;
			double[] lambdaKept = lambdaAll.Skip(0).Take(numberOfEigs).ToArray();
			double[,] EigenvectorsAll = evd.Eigenvectors;
			double[,] EigenvectorsKept = new double[EigenvectorsAll.GetLength(0), numberOfEigs];
			for (var i = 0; i < numberOfEigs; i++)
			{
				for (var j = 0; j < numberOfEigs; j++)
				{
					EigenvectorsKept[i, j] = EigenvectorsAll[i,j];
				}
			}

			if (scaled == true)
			{
				maxEigVal = lambdaKept[0];
				for (var j = 0; j < numberOfEigs; j++)
				{
					lambdaKept[j] = lambdaKept[j] / maxEigVal;
					tempSum = 0;
					for (var i = 0; i < EigenvectorsKept.GetLength(0); i++)
					{
						tempSum += Math.Pow(EigenvectorsKept[i, j],2);
					}

					for (var i = 0; i < EigenvectorsKept.GetLength(0); i++)
					{
						EigenvectorsKept[i, j] = EigenvectorsKept[i, j]/Math.Sqrt(tempSum);
					}
				}
			}

			return (lambdaKept, EigenvectorsKept);

		}

	}
}
