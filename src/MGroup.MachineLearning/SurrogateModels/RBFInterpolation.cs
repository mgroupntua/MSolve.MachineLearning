using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using MGroup.LinearAlgebra.Matrices;
using MGroup.LinearAlgebra.Vectors;
using MGroup.MachineLearning.UtilityFunctions;

namespace MGroup.MachineLearning.SurrogateModels
{
	public class RBFInterpolation
	{
		private readonly int numberOfNN;

		public RBFInterpolation(int numberOfNN)
		{
			this.numberOfNN = numberOfNN;
		}


		public static double[] GaussianRBFInterpolation(double[] newCoordinateValues, double[,] coordinates, double[,] interpolantValues, int numberOfNN, double shapeParameter)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];
			double[,] coordinatesOfNNs = new double[coordinates.GetLength(0), numberOfNN];

			int[] indices = NearestNeighborSearch.KnnSearch(coordinates, newCoordinateValues, numberOfNN).Item1;
			for (int j = 0; j < numberOfNN; j++)
			{
				for (int i = 0; i < coordinates.GetLength(0); i++)
				{
					coordinatesOfNNs[i, j] = coordinates[i, indices[j]];
				}
			}

			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			for (int i = 0; i < numberOfNN; i++)
			{
				double[] iNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(0)).Select(x => coordinatesOfNNs[x, i]).ToArray();
				double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
				l[i] = Math.Exp(-Math.Pow(iDist*shapeParameter, 2));
				for (int j = 0; j < numberOfNN; j++)
				{
					double[] jNearestNeighbor = Enumerable.Range(0, coordinates.GetLength(1)).Select(x => coordinates[j, x]).ToArray();
					double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
					L[i, j] = Math.Exp(-Math.Pow(ijDist * shapeParameter, 2));
				}
			}

			Vector lVector = Vector.CreateFromArray(l);
			Matrix matrixOfL = Matrix.CreateFromArray(L);
			Matrix invL = matrixOfL.Invert();
			Matrix matrixOfCoordinatesOfNNs = Matrix.CreateFromArray(coordinatesOfNNs);
			Matrix A = invL.MultiplyRight(matrixOfCoordinatesOfNNs.Transpose());
			Matrix Atranspose = A.Transpose();
			Vector newinterpolantValues= Atranspose.Multiply(lVector) ;

			return newInterpolantValues;
		}

		public static double[] PolyharmonicSplineInterpolation(double[] newCoordinateValues, double[,] coordinates, double[,] interpolantValues, int numberOfNN, int order)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];
			double[,] coordinatesOfNNs = new double[coordinates.GetLength(0), numberOfNN];

			int[] indices = NearestNeighborSearch.KnnSearch(coordinates, newCoordinateValues, numberOfNN).Item1;
			for (int j = 0; j < numberOfNN; j++)
			{
				for (int i = 0; i < coordinates.GetLength(0); i++)
				{
					coordinatesOfNNs[i, j] = coordinates[i, indices[j]];
				}
			}

			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			if (order % 2==0)
			{
				for (int i = 0; i < numberOfNN; i++)
				{
					double[] iNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[i, x]).ToArray();
					double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
					l[i] = Math.Pow(iDist, (2 * order) - 1);
					for (int j = 0; j < numberOfNN; j++)
					{
						double[] jNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[j, x]).ToArray();
						double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
						L[i,j]= Math.Pow(ijDist, (2 * order)- 1);
					}
				}
			}
			else
			{
				for (int i = 0; i < numberOfNN; i++)
				{
					double[] iNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[i, x]).ToArray();
					double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
					l[i] = Math.Pow(iDist, 2 * (order-1))*Math.Log(iDist);
					for (int j = 0; j < numberOfNN; j++)
					{
						double[] jNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[j, x]).ToArray();
						double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
						L[i, j] = Math.Pow(iDist, 2 * (order - 1)) * Math.Log(iDist);
					}
				}
			}

			Vector lVector = Vector.CreateFromArray(l);
			Matrix matrixOfL = Matrix.CreateFromArray(L);
			Matrix invL = matrixOfL.Invert();
			Matrix matrixOfCoordinatesOfNNs = Matrix.CreateFromArray(coordinatesOfNNs);
			Matrix A = invL.MultiplyRight(matrixOfCoordinatesOfNNs.Transpose());
			Matrix Atranspose = A.Transpose();
			Vector newinterpolantValues = Atranspose.Multiply(lVector);

			return newInterpolantValues;
		}


		public static double[] MultiquadricsRBFInterpolation(double[] newCoordinateValues, double[,] coordinates, double[,] interpolantValues, int numberOfNN, double shapeParameter)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];
			double[,] coordinatesOfNNs = new double[coordinates.GetLength(0), numberOfNN];

			int[] indices = NearestNeighborSearch.KnnSearch(coordinates, newCoordinateValues, numberOfNN).Item1;
			for (int j = 0; j < numberOfNN; j++)
			{
				for (int i = 0; i < coordinates.GetLength(0); i++)
				{
					coordinatesOfNNs[i, j] = coordinates[i, indices[j]];
				}
			}

			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			for (int i = 0; i < numberOfNN; i++)
			{
				double[] iNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[i, x]).ToArray();
				double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
				l[i] = Math.Sqrt(Math.Pow(iDist, 2 )+ Math.Pow(shapeParameter, 2));
				for (int j = 0; j < numberOfNN; j++)
				{
						double[] jNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[j, x]).ToArray();
						double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
						L[i, j] = Math.Sqrt(Math.Pow(ijDist, 2) + Math.Pow(shapeParameter, 2));
				}
			}

			Vector lVector = Vector.CreateFromArray(l);
			Matrix matrixOfL = Matrix.CreateFromArray(L);
			Matrix invL = matrixOfL.Invert();
			Matrix matrixOfCoordinatesOfNNs = Matrix.CreateFromArray(coordinatesOfNNs);
			Matrix A = invL.MultiplyRight(matrixOfCoordinatesOfNNs.Transpose());
			Matrix Atranspose = A.Transpose();
			Vector newinterpolantValues = Atranspose.Multiply(lVector);

			return newInterpolantValues;
		}

		public static double[] InverseMultiquadricsRBFInterpolation(double[] newCoordinateValues, double[,] coordinates, double[,] interpolantValues, int numberOfNN, double shapeParameter)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];
			double[,] coordinatesOfNNs = new double[coordinates.GetLength(0), numberOfNN];

			int[] indices = NearestNeighborSearch.KnnSearch(coordinates, newCoordinateValues, numberOfNN).Item1;
			for (int j = 0; j < numberOfNN; j++)
			{
				for (int i = 0; i < coordinates.GetLength(0); i++)
				{
					coordinatesOfNNs[i, j] = coordinates[i, indices[j]];
				}
			}

			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			for (int i = 0; i < numberOfNN; i++)
			{
				double[] iNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[i, x]).ToArray();
				double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
				l[i] = 1/Math.Sqrt(Math.Pow(iDist, 2) + Math.Pow(shapeParameter, 2));
				for (int j = 0; j < numberOfNN; j++)
				{
					double[] jNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[j, x]).ToArray();
					double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
					L[i, j] = 1/Math.Sqrt(Math.Pow(ijDist, 2) + Math.Pow(shapeParameter, 2));
				}
			}

			Vector lVector = Vector.CreateFromArray(l);
			Matrix matrixOfL = Matrix.CreateFromArray(L);
			Matrix invL = matrixOfL.Invert();
			Matrix matrixOfCoordinatesOfNNs = Matrix.CreateFromArray(coordinatesOfNNs);
			Matrix A = invL.MultiplyRight(matrixOfCoordinatesOfNNs.Transpose());
			Matrix Atranspose = A.Transpose();
			Vector newinterpolantValues = Atranspose.Multiply(lVector);

			return newInterpolantValues;
		}

		public static double[] InverseQuadraticsRBFInterpolation(double[] newCoordinateValues, double[,] coordinates, double[,] interpolantValues, int numberOfNN, double shapeParameter)
		{
			double[] newInterpolantValues = new double[interpolantValues.GetLength(0)];
			double[,] coordinatesOfNNs = new double[coordinates.GetLength(0), numberOfNN];

			int[] indices = NearestNeighborSearch.KnnSearch(coordinates, newCoordinateValues, numberOfNN).Item1;
			for (int j = 0; j < numberOfNN; j++)
			{
				for (int i = 0; i < coordinates.GetLength(0); i++)
				{
					coordinatesOfNNs[i, j] = coordinates[i, indices[j]];
				}
			}

			double[] l = new double[numberOfNN];
			double[,] L = new double[numberOfNN, numberOfNN];

			for (int i = 0; i < numberOfNN; i++)
			{
				double[] iNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[i, x]).ToArray();
				double iDist = GetDistance(newCoordinateValues, iNearestNeighbor);
				l[i] = 1 /(Math.Pow(iDist, 2) + Math.Pow(shapeParameter, 2));
				for (int j = 0; j < numberOfNN; j++)
				{
					double[] jNearestNeighbor = Enumerable.Range(0, coordinatesOfNNs.GetLength(1)).Select(x => coordinatesOfNNs[j, x]).ToArray();
					double ijDist = GetDistance(iNearestNeighbor, jNearestNeighbor);
					L[i, j] = 1 / (Math.Pow(ijDist, 2) + Math.Pow(shapeParameter, 2));
				}
			}

			Vector lVector = Vector.CreateFromArray(l);
			Matrix matrixOfL = Matrix.CreateFromArray(L);
			Matrix invL = matrixOfL.Invert();
			Matrix matrixOfCoordinatesOfNNs = Matrix.CreateFromArray(coordinatesOfNNs);
			Matrix A = invL.MultiplyRight(matrixOfCoordinatesOfNNs.Transpose());
			Matrix Atranspose = A.Transpose();
			Vector newinterpolantValues = Atranspose.Multiply(lVector);

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
