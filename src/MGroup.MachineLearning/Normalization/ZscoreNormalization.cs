using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning.Normalization
{
	/// <summary>
	/// Normalize the data using the
	/// Z-score normalization so that
	/// their values have zero mean and unit variance.
	/// </summary>
	public class ZscoreNormalization : INormalization
	{
		public (double[,], double[], double[]) Normalize(double[,] rawData)
		{
			double[,] scaledData = new double[rawData.GetLength(0), rawData.GetLength(1)];
			double[] meanValuePerRow = new double[rawData.GetLength(0)];
			double[] stdValuePerRow = new double[rawData.GetLength(0)];
			for (int row = 0; row < rawData.GetLength(0); row++)
			{
				for (int col = 0; col < rawData.GetLength(1); col++)
				{
					meanValuePerRow[row] += rawData[row, col];
				}
				meanValuePerRow[row] = meanValuePerRow[row] / rawData.GetLength(1);
			}

			for (int row = 0; row < rawData.GetLength(0); row++)
			{
				for (int col = 0; col < rawData.GetLength(1); col++)
				{
					stdValuePerRow[row] += Math.Pow(rawData[row, col]- meanValuePerRow[row],2);
				}
				stdValuePerRow[row] = Math.Sqrt(stdValuePerRow[row] / (rawData.GetLength(1)-1));
			}

			for (int row = 0; row < rawData.GetLength(0); row++)
			{
				for (int col = 0; col < rawData.GetLength(1); col++)
				{
					scaledData[row, col] = (rawData[row, col] - meanValuePerRow[row]) / stdValuePerRow[row];
				}
			}

			return (scaledData, meanValuePerRow, stdValuePerRow);
		}
	}
}
