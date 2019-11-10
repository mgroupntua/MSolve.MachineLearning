using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MGroup.MachineLearning.UtilityFunctions;

namespace MGroup.MachineLearning.SurrogateModels
{
	class RBFInterpolation
	{
		private readonly int numberOfNN;

		public RBFInterpolation(int numberOfNN)
		{
			this.numberOfNN = numberOfNN;
		}


		public static double[] GaussianRBFInterpolation(double[] newCoordinateValues, double[,] Coordinates, double[,] interpolantValues, int numberOfNN, double shapeParameter)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];

			int[,] indices = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item1;
			double[,] sortedDistances = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item2;
			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			for (int i = 0; i < numberOfNN; i++)
			{
				double[] iNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[i, x]).ToArray();
				double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
				l[i] = Math.Exp(-Math.Pow(iDist*shapeParameter, 2));
				for (int j = 0; j < numberOfNN; j++)
				{
					double[] jNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[j, x]).ToArray();
					double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
					L[i, j] = Math.Exp(-Math.Pow(ijDist * shapeParameter, 2));
				}
			}

			//A=L\Umatrix(:,id)';
			//newinterpolantValues=A'*l';

			return newInterpolantValues;
		}

		public static double[] PolyharmonicSplineInterpolation(double[] newCoordinateValues, double[,] Coordinates, double[,] interpolantValues, int numberOfNN, int order)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];

			int[,] indices = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item1;
			double[,] sortedDistances = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item2;
			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN,numberOfNN];

			if (order % 2==0)
			{
				for (int i = 0; i < numberOfNN; i++)
				{
					double[] iNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[i, x]).ToArray();
					double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
					l[i] = Math.Pow(iDist, (2 * order) - 1);
					for (int j = 0; j < numberOfNN; j++)
					{
						double[] jNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[j, x]).ToArray();
						double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
						L[i,j]= Math.Pow(ijDist, (2 * order)- 1);
					}
				}
			}
			else
			{
				for (int i = 0; i < numberOfNN; i++)
				{
					double[] iNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[i, x]).ToArray();
					double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
					l[i] = Math.Pow(iDist, 2 * (order-1))*Math.Log(iDist);
					for (int j = 0; j < numberOfNN; j++)
					{
						double[] jNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[j, x]).ToArray();
						double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
						L[i, j] = Math.Pow(iDist, 2 * (order - 1)) * Math.Log(iDist);
					}
				}
			}

			//A=L\Umatrix(:,id)';
			//newinterpolantValues=A'*l';

			return newInterpolantValues;
		}


		public static double[] MultiquadricsRBFInterpolation(double[] newCoordinateValues, double[,] Coordinates, double[,] interpolantValues, int numberOfNN, double shapeParameter)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];

			int[,] indices = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item1;
			double[,] sortedDistances = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item2;
			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			for (int i = 0; i < numberOfNN; i++)
			{
				double[] iNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[i, x]).ToArray();
				double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
				l[i] = Math.Sqrt(Math.Pow(iDist, 2 )+ Math.Pow(shapeParameter, 2));
				for (int j = 0; j < numberOfNN; j++)
				{
						double[] jNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[j, x]).ToArray();
						double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
						L[i, j] = Math.Sqrt(Math.Pow(ijDist, 2) + Math.Pow(shapeParameter, 2));
				}
			}

			//A=L\Umatrix(:,id)';
			//newinterpolantValues=A'*l';

			return newInterpolantValues;
		}

		public static double[] InverseMultiquadricsRBFInterpolation(double[] newCoordinateValues, double[,] Coordinates, double[,] interpolantValues, int numberOfNN, double shapeParameter)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];

			int[,] indices = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item1;
			double[,] sortedDistances = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item2;
			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			for (int i = 0; i < numberOfNN; i++)
			{
				double[] iNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[i, x]).ToArray();
				double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
				l[i] = 1/Math.Sqrt(Math.Pow(iDist, 2) + Math.Pow(shapeParameter, 2));
				for (int j = 0; j < numberOfNN; j++)
				{
					double[] jNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[j, x]).ToArray();
					double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
					L[i, j] = 1/Math.Sqrt(Math.Pow(ijDist, 2) + Math.Pow(shapeParameter, 2));
				}
			}

			//A=L\Umatrix(:,id)';
			//newinterpolantValues=A'*l';

			return newInterpolantValues;
		}

		public static double[] InverseQuadraticsRBFInterpolation(double[] newCoordinateValues, double[,] Coordinates, double[,] interpolantValues, int numberOfNN, double shapeParameter)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];

			int[,] indices = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item1;
			double[,] sortedDistances = NearestNeighborSearch.KnnSearch(Coordinates, Coordinates, numberOfNN).Item2;
			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			for (int i = 0; i < numberOfNN; i++)
			{
				double[] iNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[i, x]).ToArray();
				double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
				l[i] = 1 /(Math.Pow(iDist, 2) + Math.Pow(shapeParameter, 2));
				for (int j = 0; j < numberOfNN; j++)
				{
					double[] jNearestNeighbor = Enumerable.Range(0, Coordinates.GetLength(1)).Select(x => Coordinates[j, x]).ToArray();
					double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
					L[i, j] = 1 / (Math.Pow(ijDist, 2) + Math.Pow(shapeParameter, 2));
				}
			}

			//A=L\Umatrix(:,id)';
			//newinterpolantValues=A'*l';

			return newInterpolantValues;
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
