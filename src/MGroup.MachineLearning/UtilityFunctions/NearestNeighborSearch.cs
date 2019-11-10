using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MGroup.MachineLearning.UtilityFunctions
{
	public class NearestNeighborSearch
	{
		public static (int[,], double[,]) KnnSearch(double[,] trainSamples, double[,] testSamples, int K)
		{

			var indices = new int[testSamples.GetLength(0), testSamples.GetLength(0)];
			var testNumber = testSamples.GetLength(0);
			var trainNumber = trainSamples.GetLength(0);
			// Declaring these here so that I don't have to 'new' them over and over again in the main loop, 
			// just to save some overhead
			var distances = new double[trainNumber][];
			for (var i = 0; i < trainNumber; i++)
			{
				distances[i] = new double[2]; // Will store both distance and index in here
			}

			// Performing KNN ...
			for (int tst = 0; tst < testNumber; tst++)
			{
				// For every test sample, calculate distance from every training sample
				for (var trn = 0; trn < trainNumber; trn++)
				{
					double[] testSample = Enumerable.Range(0, testSamples.GetLength(1)).Select(x => testSamples[tst, x]).ToArray();
					double[] trainSample = Enumerable.Range(0, trainSamples.GetLength(1)).Select(x => trainSamples[tst, x]).ToArray();

					double dist = GetDistance(testSample, trainSample);
					// Storing distance as well as index 
					distances[trn][0] = dist;
					distances[trn][1] = trn;
				};

				// Sort distances and take top K (?What happens in case of multiple points at the same distance?)
				var votingDistances = distances.AsParallel().OrderBy(t => t[0]).Take(K);
			}

			return (indices, sortedDistances);
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
