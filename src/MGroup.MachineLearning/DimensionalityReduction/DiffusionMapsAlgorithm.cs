using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Accord.Math;
using MGroup.MachineLearning.UtilityFunctions;


namespace MGroup.MachineLearning.DimensionalityReduction
{
	public class DiffusionMapsAlgorithm
	{
		// DMAP algorithm parameters
		private int numberOfKNN;
		private int NNofKDE;
		private int differentialOperator;  // 1:Laplace-Beltrami, 2:generator of grad system
		private int numberOfEigenvectors;
		private double[,] dataSet;
		double alpha, beta;

		// read the matrix containing the problem data
		public DiffusionMapsAlgorithm(double[,] dataSet, int numberOfKNN, int NNofKDE, int differentialOperator, int numberOfEigenvectors)
		{
			this.dataSet = dataSet;
			this.numberOfKNN = numberOfKNN;
			this.NNofKDE = NNofKDE;
			this.differentialOperator = differentialOperator;
			this.numberOfEigenvectors = numberOfEigenvectors;
		}

		public void ProcessData()
		{
			int N = dataSet.GetLength(0);
			int[,] indices = new int[N, numberOfKNN];
			double[,] sortedDistances= new double[N, numberOfKNN];
			for (var i = 0; i < N; i++)
			{
				double[] dataPoint = Enumerable.Range(0, dataSet.GetLength(1)).Select(x => dataSet[i, x]).ToArray();
				int[] ind = NearestNeighborSearch.KnnSearch(dataSet, dataPoint, numberOfKNN).Item1;
				double[] sortedDist = NearestNeighborSearch.KnnSearch(dataSet, dataPoint, numberOfKNN).Item2;
				for (var j = 0; j < numberOfKNN; j++)
				{
					indices[i, j] = ind[j];
					sortedDistances[i, j] = sortedDist[j];
				}
			}

			// build ad hoc bandwidth function by autotuning epsilon for each point
			double[] epss = new double[401];
			for (var i = 0; i < 401; i++)
			{
				epss[i] = Math.Pow(2, -30+i * 0.1); // Will store both distance and index in here
			}
			double[] rho0 = evaluateRho0(sortedDistances, NNofKDE);
			// pre-kernel used with ad hoc bandwidth only for estimating dimension and sampling density
			double[,] dt = evaluateDt(sortedDistances, rho0, indices, numberOfKNN);
			// tune epsilon on the pre-kernel
			double[] dpreGlobal = evaluateDpreGlobal(epss, dt, numberOfKNN, N);
			(double maxval, int maxind) = findMaxvalMaxind(dpreGlobal, epss);
			double dim = 2 * maxval;

			// use ad hoc bandwidth function, rho0, to estimate the density
			for (var i = 0; i < dt.GetLength(0); i++)
			{
				for (var j = 0; j < dt.GetLength(1); j++)
				{
					dt[i, j] = Math.Exp(-dt[i, j] / (2 * epss[maxind])) / Math.Pow(2 * Math.PI * epss[maxind], (dim / 2));
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
					sum += reshapedDt[i, j];
				}
				qest[i] = sum / (N * Math.Pow(rho0[i], dim));
			}

			if (differentialOperator == 1)
			{
				beta = -0.5;
				alpha = -dim / 4 +0.5;
			}
			else if (differentialOperator == 2)
			{
				beta = -0.5;
				alpha = -dim / 4;
			}

			double c1 = 2 - (2 * alpha) + (dim * beta) + (2 * beta);
			double c2 = 0.5 - (2 * alpha) + (2 * dim * alpha) + (dim * beta / 2) + beta;

			for (var i = 0; i < sortedDistances.GetLength(0); i++)
			{
				for (int j = 0; j < sortedDistances.GetLength(1); j++)
				{
					sortedDistances[i, j] = Math.Pow(sortedDistances[i, j], 2);
				}
			}

			// construct bandwidth function rho(x) from the sampling density estimate
			double[] rho = new double[qest.Length];
			double sumRho = 0;
			for (var i = 0; i < qest.Length; i++)
			{
				rho[i] = Math.Pow(qest[i], beta);
				sumRho += rho[i];
			}
			for (var i = 1; i < qest.Length; i++)
			{
				rho[i] = rho[i] / sumRho * rho.Length;
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
					sortedDistances[i, j] = Math.Exp(-sortedDistances[i, j] / (4 * epsilon));
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

			double temp;
			// q^S_epsilon (this is the sampling density estimate q(x) obtained from the VB kernel)
			for (var i = 0; i < symmetricSortedDistances.GetLength(0); i++)
			{
				temp = 0;
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

			double[,] DtimesDinv = new double[N, N];
			DtimesDinv = Accord.Math.Matrix.Dot(symmetricSortedDistances, Dinv1);
			double[,] newD = new double[N, N];
			newD = Accord.Math.Matrix.Dot(Dinv1, DtimesDinv);

			double[,] Sinv = new double[N, N];
			double temp2;
			for (var i = 0; i < N; i++)
			{
				temp2 = 0;
				for (var j = 0; j < N; j++)
				{
					temp2 += newD[i,j];
				}
				Sinv[i, i] = Math.Pow(Math.Pow(rho[i], 2) * temp2, -0.5);
			}

			double[,] newDtimesSinv = new double[N, N];
			newDtimesSinv = Accord.Math.Matrix.Dot(newD, Sinv);
			double[,] finalD = new double[N, N];
			finalD = Accord.Math.Matrix.Dot(Sinv, newDtimesSinv);
			for (var i = 0; i < N; i++)
			{
				finalD[i, i] = finalD[i, i] - Math.Pow(rho[i], -2) +1;
			}

			double[] DMAPEigVals;
			double[,] DMAPEigVecs;
			bool sort = true;
			bool inPlace = false;
			bool scaled = true;
			(DMAPEigVals, DMAPEigVecs) = EigenDecomposition.FindEigenValuesAndEigenvectorsSymmetricOnly(finalD, numberOfEigenvectors, inPlace, sort, scaled);
			for (var i = 0; i < numberOfEigenvectors; i++)
			{
				DMAPEigVals[i] = Math.Log(DMAPEigVals[i])/epsilon;
			}
			DMAPEigVecs = Accord.Math.Matrix.Dot(Sinv, DMAPEigVecs);


			// normalize qest into a density
			for (var i = 0; i <N; i++)
			{
				qest[i] = qest[i]/(N*Math.Pow(4*Math.PI*epsilon,dim/2));
			}

			// constuct the invariant measure of the system
			double[] peq = new double[N];
			for (var i = 0; i < N; i++)
			{
				peq[i] = qest[i] * Math.Pow(Sinv[i,i], -2);
			}

			double normalizationFactor=0;
			for (var i = 0; i < N; i++)
			{
				normalizationFactor += peq[i]/qest[i]/N;
			}

			// normalize the invariant measure
			for (var i = 0; i < N; i++)
			{
				peq[i] = peq[i] / normalizationFactor;
			}

			//normalize eigenfunctions such that: \sum_i psi(x_i)^2 p(x_i)/q(x_i) = 1
			double[] EigVec_i = new double[N];
			double[]tempVector = new double[N];
			double meanTempVector = 0;
			for (var i = 0; i < numberOfEigenvectors; i++)
			{
				EigVec_i= Enumerable.Range(0, DMAPEigVecs.GetLength(0)).Select(x => DMAPEigVecs[x, i]).ToArray();
				for (var j = 0; j < N; j++)
				{

					tempVector[j]=Math.Pow(EigVec_i[j],2)*(peq[j]/qest[j]);
				}
				meanTempVector = FindMeanOfVector(tempVector);
				for (var j = 0; j < N; j++)
				{

					DMAPEigVecs[j,i] = DMAPEigVecs[j, i]/Math.Sqrt(meanTempVector);
				}
			}
		}


		private static double[] evaluateRho0(double[,] dist, int k)
		{
			double[,] distNew = new double[dist.GetLength(0), k - 1];
			for (var i = 0; i < distNew.GetLength(0); i++)
			{
				for (var j = 0; j < k - 1; j++)
				{
					distNew[i, j] = Math.Pow(dist[i, j + 1], 2);
				}
			}

			double[] rho0 = new double[dist.GetLength(0)];
			for (var i = 0; i < distNew.GetLength(0); i++)
			{
				for (var j = 0; j < distNew.GetLength(1); j++)
				{
					rho0[i] += distNew[i,j];
				}
				rho0[i] = Math.Sqrt(rho0[i] / distNew.GetLength(1));
			}
			return rho0;
		}

		private static double FindMeanOfVector(double[] vec)
		{
			double mean= 0;
			int N = vec.Length;
			for (var i = 0; i < N; i++)
			{
				mean += vec[i] / N;
			}
			return mean;
		}

		private static double[,] evaluateDt(double[,] dist, double[] rho0, int[,] indices, int k)
		{
			double[,] temp = new double[dist.GetLength(0), k];
			double[,] dt = new double[dist.GetLength(0), k];
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

			return dt;

		}

		private static double[] evaluateDpreGlobal(double[] epss, double[,] dt, int k, int N)
		{
			double[,] temp = new double[dt.GetLength(0), dt.GetLength(1)];
			double[] dpreGlobal = new double[epss.Length];
			for (var i = 0; i < epss.Length; i++)
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

		private static (double, int) findMaxvalMaxind(double[] dpreGlobal, double[] epss)
		{
			double[] temp1 = new double[dpreGlobal.Length - 1];
			double[] temp2 = new double[epss.Length - 1];
			double[] temp3 = new double[epss.Length - 1];
			for (var i = 0; i < dpreGlobal.Length - 1; i++)
			{
				temp1[i] = Math.Log(dpreGlobal[i + 1]) - Math.Log(dpreGlobal[i]);
				temp2[i] = Math.Log(epss[i + 1]) - Math.Log(epss[i]);
				temp3[i] = temp1[i] / temp2[i];
			}
			double maxval = temp3.Max();
			int maxind = temp3.ToList().IndexOf(maxval);

			return (maxval, maxind);
		}

		private static double[,] reshapeDt(double[,] dt, int[,] indices, int N, int k)
		{
			double[,] reshapedDt = new double[N, N];
			for (var i = 0; i < N; i++)
			{
				for (var j = 0; j < k; j++)
				{
					reshapedDt[i, indices[i, j]] = dt[i, j];
				}
			}
			return reshapedDt;
		}

		private static double evaluateFinalEpsilon(double[] epss, double[,] sortedDistances, int N, int numberOfKNN)
		{
			double[] s = new double[epss.Length];
			for (var i = 0; i < epss.Length; i++)
			{
				double[,] temp = new double[sortedDistances.GetLength(0), sortedDistances.GetLength(1)];
				double tempSum = 0;
				for (var j = 0; j < sortedDistances.GetLength(0); j++)
				{
					for (var q = 0; q < sortedDistances.GetLength(1); q++)
					{
						temp[j, q] = Math.Exp(-sortedDistances[j, q] / (4 * epss[i]));
						tempSum += temp[j, q];
					}
				}
				s[i] = tempSum / (N * numberOfKNN);
			}

			double[] temp2 = new double[epss.Length];
			for (var i = 0; i < epss.Length - 1; i++)
			{
				temp2[i] = (Math.Log(s[i + 1]) - Math.Log(s[i])) / (Math.Log(epss[i + 1]) - Math.Log(epss[i]));
			}

			double maxval = temp2.Max();
			int maxind = temp2.ToList().IndexOf(maxval);

			double epsilon = epss[maxind];

			return epsilon;
		}
	}
}
