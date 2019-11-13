using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MGroup.MachineLearning.UtilityFunctions
{
	public class NearestNeighborSearch
	{
		public static (int[], double[]) KnnSearch(double[,] trainSamples, double[] testSample, int K)
		{
			int trainNumber = trainSamples.GetLength(1);
			var distances = new double[trainNumber][];
			for (var i = 0; i < trainNumber; i++)
			{
				distances[i] = new double[2]; // Will store both distance and index in here
			}

			// For every test sample, calculate distance from every training sample
			for (var trn = 0; trn < trainNumber; trn++)
			{
				double[] trainSample = Enumerable.Range(0, trainSamples.GetLength(1)).Select(x => trainSamples[trn, x]).ToArray();

				double dist = GetDistance(testSample, trainSample);
				// Storing distance as well as index 
				distances[trn][0]= dist;
				distances[trn][1] = trn;
			};

			// Sort distances and take top K (?What happens in case of multiple points at the same distance?)
			var sortedDistancesAndIndices = distances.AsParallel().OrderBy(t => t[0]).Take(K);
			double[] sortedDistances = sortedDistancesAndIndices.AsEnumerable().Select(t=>t[0]).ToArray();
			int[] sortedIndices =  sortedDistancesAndIndices.AsEnumerable().Select(t => (int) t[1]).ToArray();
			return (sortedIndices, sortedDistances);
		}

		private static double GetDistance(double[] sample1, double[] sample2)
		{
			var distance = 0.0;
			// assume sample1 and sample2 are valid i.e. same length 

			for (var i = 0; i < sample1.Length; i++)
			{
				var temp = sample1[i] - sample2[i];
				distance += temp * temp;
			}
			distance = Math.Sqrt(distance);
			return distance;
		}
	}
}
