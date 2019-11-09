using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace MGroup.MachineLearning.DimensionalityReduction
{
	public class DiffusionMapsAlgorithm
	{
		// DMAP algorithm parameters
		private int numberOfKNN = 256;
		private int NNofKDE = 64;
		private int differentialOperator = 1;  // 1:Laplace-Beltrami, 2:generator of grad system
		private int numberOfEigenvectors = 3;
		private double[,] dataSet = DiffusionMapsAlgorithm.ImportData();

		// read the matrix containing the problem data
		private static double[,] ImportData()
		{
			var filePath = @"E:\GIANNIS_DATA\DESKTOP\VS\trainingPoints.csv";
			string[][] dataValues = File.ReadLines(filePath).Select(x => x.Split(',')).ToArray();
			double[,] dataSet = new double[dataValues.GetLength(0), dataValues.GetLength(1)]; 

			for (int i = 0; i < dataValues.GetLength(0); i++)
			{
				for (int j = 0; j < dataValues.GetLength(1); j++)
				{
					dataSet[i,j]= Convert.ToDouble(dataValues[i][j]);
				}
			}

			return dataSet;
		}
		private DiffusionMapsAlgorithm(double[,] dataSet, int numberOfKNN, int NNofKDE, int differentialOperator, int numberOfEigenvectors)
		{
			this.dataSet = dataSet;
			this.numberOfKNN = numberOfKNN;
			this.NNofKDE = NNofKDE;
			this.differentialOperator = differentialOperator;
			this.numberOfEigenvectors = numberOfEigenvectors;
		}

		private void ProcessData()
		{
			int N = dataSet.GetLength(0);
			int[,] indices = KnnSearch(dataSet, dataSet, numberOfKNN).Item1;
			double[,] sortedDistances = KnnSearch(dataSet, dataSet, numberOfKNN).Item2;

			// build ad hoc bandwidth function by autotuning epsilon for each point
			double[] epss = new double[401];
			for (var i = 0; i < 401; i++)
			{
				epss[i] = Math.Pow(2, i * 0.01); // Will store both distance and index in here
			}
			double[] rho0 = evaluateRho0(sortedDistances, NNofKDE);
			// pre-kernel used with ad hoc bandwidth only for estimating dimension and sampling density
			double[,] dt = evaluateDt(rho0, indices, numberOfKNN);
			// tune epsilon on the pre-kernel
			double[] dpreGlobal = evaluateDpreGlobal(epss, dt, numberOfKNN, N);
			double maxval = findMaxvalMaxind().Item1;
			double maxind = findMaxvalMaxind().Item2;
			double dim = 2 * maxval;

			// use ad hoc bandwidth function, rho0, to estimate the density
			for (var i = 0; i < dt.GetLength(0); i++)
			{
				for (var j = 0; j < dt.GetLength(1); j++)
				{
					dt[i, j] = Math.Exp(-dt[i, j] / (2 * epss[maxind])) / Math.pow(2 * Math.PI * epss[maxind], (dim / 2));
				}
			}

			// the matrix created here might be large, must change it to sparse format
			double[,] reshapedDt = reshapeDt(dt, indices, N, numberOfKNN);
			double[,] symmetricDt = new double[reshapedDt.GetLength(0), reshapedDt.GetLength(1)];
			for (var i = 0; i < reshapedDt.GetLength(0); i++)
			{
				for (var j = 0; j < reshapedDt.GetLength(1); j++)
				{
					symmetricDt[i, j] = (reshapedDt[i, j] + reshapedDt[j, i]) / 2;
				}
			}

			// sampling density estimate for bandwidth function
			double[] qest = new double[symmetricDt.GetLength(0)];
			for (var i = 0; i < symmetricDt.GetLength(0); i++)
			{
				double sum = 0;
				for (var j = 0; j < symmetricDt.GetLength(1); j++)
				{
					sum += dt[i, j];
				}
				qest[i] = sum / (N * Math.Pow(rho0[i], dim));
			}

			if (differentialOperator == 1)
			{
				double beta = -1 / 2;
				double alpha = -dim / 4 + 1 / 2;
			}
			else if (differentialOperator == 2)
			{
				double beta = -1 / 2;
				double alpha = -dim / 4;
			}

			double c1 = 2 - 2 * alpha + dim * beta + 2 * beta;
			double c2 = 0.5 - 2 * alpha + 2 * dim * alpha + dim * beta / 2 + beta;

			for (var i = 0; i < sortedDistances.GetLength(0); i++)
			{
				for (var j = 0; j < sortedDistances.GetLength(1); j++)
				{
					sortedDistances[i, j] = Math.Exp(sortedDistances[i, j], 2);
				}
			}

			// construct bandwidth function rho(x) from the sampling density estimate
			double[] rho = new double[qest.GetLength()];
			double sumRho = 0;
			for (var i = 1; i < qest.GetLength(); i++)
			{
				rho[i] = Math.Pow(qest[i], beta);
				sumRho += rho[i];
			}
			for (var i = 1; i < qest.GetLength(); i++)
			{
				rho[i] = rho[i] / sumRho * rho.GetLength();
			}

			// construct the exponent of K^S_epsilon
			for (var i = 0; i < sortedDistances.GetLength(0); i++)
			{
				for (var j = 0; j < sortedDistances.GetLength(1); j++)
				{
					sortedDistances[i, j] = sortedDistances[i, j] / rho[i];
				}
			}
			for (var i = 0; i < sortedDistances.GetLength(0); i++)
			{
				for (var j = 0; j < sortedDistances.GetLength(1); j++)
				{
					sortedDistances[i, j] = sortedDistances[i, j] / rho[indices[i, j]];
				}
			}

			// tune epsilon for the final kernel
			double epsilon = evaluateFinalEpsilon(epss, sortedDistances, N, numberOfKNN);

			// K^S_epsilon with final choice of epsilon
			for (var i = 0; i < sortedDistances.GetLength(0); i++)
			{
				for (var j = 0; j < sortedDistances.GetLength(1); j++)
				{
					sortedDistances[i, j] = Math.Exp(sortedDistances[i, j] / 4 * epsilon);
				}
			}

			// the matrix created here might be large, must change it to sparse format
			double[,] reshapedSortedDistances = reshapeDt(sortedDistances, indices, N, numberOfKNN);
			double[,] symmetricSortedDistances = new double[reshapedSortedDistances.GetLength(0), reshapedSortedDistances.GetLength(1)];
			for (var i = 0; i < reshapedSortedDistances.GetLength(0); i++)
			{
				for (var j = 0; j < reshapedSortedDistances.GetLength(1); j++)
				{
					symmetricSortedDistances[i, j] = (reshapedSortedDistances[i, j] + reshapedSortedDistances[j, i]) / 2;
				}
			}

			// q^S_epsilon (this is the sampling density estimate q(x) obtained from the VB kernel)
			for (var i = 0; i < symmetricSortedDistances.GetLength(0); i++)
			{
				double temp = 0;
				for (var j = 0; j < symmetricSortedDistances.GetLength(1); j++)
				{
					temp += symmetricSortedDistances[i, j];
				}
				qest[i] = temp / Math.Pow(rho[i], dim);
			}

			double[,] Dinv1 = new double[N, N];
			for (var i = 0; i < N; i++)
			{
				Dinv1[i, i] = Math.Pow(qest[i], -alpha);
			}

		}

		private static Tuple<int[,], double[,]> KnnSearch(double[,] trainSamples, double[,] testSamples, int K)
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
					var dist = GetDistance(testSamples[tst], trainSamples[trn]);
					// Storing distance as well as index 
					distances[trn, 0] = dist;
					distances[trn, 1] = trn;
				};

				// Sort distances and take top K (?What happens in case of multiple points at the same distance?)
				var votingDistances = distances.AsParallel().OrderBy(t => t[0]).Take(K);

			}

			return new Tuple<int[,], double[,]>(indices, sortedDistances);
		}

		private static double GetDistance(double[] sample1, double[] sample2)
		{
			var distance = 0.0;
			// assume sample1 and sample2 are valid i.e. same length 

			for (var i = 0; i < sample1.GetLength; i++)
			{
				var temp = sample1[i] - sample2[i];
				distance += temp * temp;
			}
			return distance;
		}

		private static double[] evaluateRho0(double[,] dist, int k)
		{
			distNew = new double[dist.GetLength(0), k - 1];
			for (var i = 0; i < distNew.GetLength(0); i++)
			{
				for (var j = 0; j < k - 1; j++)
				{
					distNew[i, j] = Math.Pow(dist[i, j + 1], 2);
				}
			}

			rho0 = new double[dist.GetLength(0)];
			for (var i = 0; i < distNew.GetLength(0); i++)
			{
				for (var j = 0; j < distNew.GetLength(1); j++)
				{
					rho0[i] += rho0[i];
				}
				rho0[i] = rho0[i] / distNew.GetLength(1);
			}
			return rho0;
		}

		private static double[,] evaluateDt(double[,] dist, double[] rho0, int[,] indices, int k)
		{
			temp = new double[dist.GetLength(0), k];
			dt = new double[dist.GetLength(0), k];
			for (var i = 0; i < dist.GetLength(0); i++)
			{
				for (var j = 0; j < k; j++)
				{
					temp[i, j] = rho0[i] * rho0[indices[i, j]];
				}
			}
			for (var i = 0; i < dist.GetLength(0); i++)
			{
				for (var j = 0; j < k; j++)
				{
					dt[i, j] = Math.Pow(dist[i, j], 2) / temp[i, j];
				}
			}

			return dt

		}

		private static double[] evaluateDpreGlobal(double[] epss, double[,] dt, int k, int N)
		{
			double[,] temp = new double[dt.GetLength(0), dt.GetLenth(1)];
			double[] dpreGlobal = new double[epss.GetLength];
			for (var i = 0; i < epss.GetLength(); i++)
			{
				double tempSum = 0;
				for (var j = 0; j < dt.GetLength(0); j++)
				{
					for (var q = 0; q < dt.GetLength(1); q++)
					{
						temp[j, q] = Math.Exp(-dt[j, q] / (2 * epss[i]));
						tempSum += temp[j, q];
					}
				}
				dpreGlobal[i] = tempSum / (N * k);
			}
			return dpreGlobal;
		}

		private static Tuple<double, int> findMaxvalMaxind(double[] dpreGlobal, double[] epss)
		{
			double[] temp1 = new double[dpreGlobal.GetLength() - 1];
			double[] temp2 = new double[epss.GetLength() - 1];
			double[] temp3 = new double[epss.GetLength() - 1];
			for (var i = 0; i < dpreGlobal.GetLength() - 1; i++)
			{
				temp1[i] = Math.Log(dpreGlobal(i + 1)) - Math.Log(dpreGlobal(i));
				temp2[i] = Math.Log(epss(i + 1)) - Math.Log(epss(i));
				temp3[i] = temp1[i] / temp2[i];
			}
			double maxval = temp3.Max();
			int maxind = temp3.ToList().IndexOf(maxval);

			return new Tuple<double, int>(maxval, maxind);
		}

		private static double[,] reshapeDt(double[,] dt, int[,] indices, int N, int k)
		{
			double[] temp1 = new double[k * N];
			int[] temp2 = new int[k * N];
			int[] temp3 = new int[N];
			for (var i = 0; i < N; i++)
			{
				temp3[i] = i;
			}

			int count = 0;
			for (var i = 0; i < N; i++)
			{
				for (var j = 0; j < k; j++)
				{
					temp1[count] = dt[i, j];
					temp2[count] = indices[i, j];
					count += 1;
				}
			}

			int[] temp4 = Enumerable.Repeat(temp3, k).ToArray();

			double[,] reshapedDt = new double[N, N];
			count = 0;
			for (var i = 0; i < N; i++)
			{
				for (var j = 0; j < N; j++)
				{
					reshapedDt[j, i] = dt[temp2[count], temp1[count]];
					count += 1;
				}
			}
			return reshapedDt;
		}

		private static double evaluateFinalEpsilon(double[] epss, double[,] sortedDistances, int N, int numberOfKNN)
		{
			double[] s = new double[epss.GetLength()];
			for (var i = 0; i < epss.GetLength(); i++)
			{
				double[,] temp = new double[sortedDistances.GetLength(0), sortedDistances.GetLength(1)];
				double tempSum = 0;
				for (var j = 0; j < sortedDistances.GetLength(0); j++)
				{
					for (var q = 0; q < sortedDistances.GetLength(1); q++)
					{
						temp[j, q] = Math.Exp[sortedDistances[j, q] / (4 * epss[i])];
						tempSum += temp[j, q];
					}
				}
				s[i] = tempSum / (N * numberOfKNN);
			}

			double[] temp2 = new double[epss.GetLength()];
			for (var i = 0; i < epss.GetLength() - 1; i++)
			{
				temp2[i] = (Math.Log(s[i + 1]) - Math.Log(s[i]) / (Math.Log(epss[i + 1]) - Math.Log(epss[i]);
			}

			double maxval = temp2.Max();
			int maxind = temp2.ToList().IndexOf(maxval);

			double epsilon = epss[maxind];

			return epsilon;
		}
	}
}
