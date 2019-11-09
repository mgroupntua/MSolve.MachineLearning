using System;
using System.Collections.Generic;
using System.Text;

namespace MGroup.MachineLearning.Normalization
{
	/// <summary>
	/// Normalize the data using the
	/// MinMax normalization so that
	/// their values lie in
	/// the [0,1] domain.
	/// </summary>
	public class MinMaxNormalization : INormalization
	{
		public (double[,], double[], double[]) Normalize(double[,] rawData)
		{
			double[,] scaledData= new double[rawData.GetLength(0), rawData.GetLength(1)];
			double[] minValuePerRow = new double[rawData.GetLength(0)];
			double[] maxValuePerRow = new double[rawData.GetLength(0)];
			for (int row = 0; row < rawData.GetLength(0); row++)
			{
				minValuePerRow[row] = double.MaxValue;
				maxValuePerRow[row] = double.MinValue;

				for (int col = 0; col < rawData.GetLength(1); col++)
				{
					if (rawData[row, col] < minValuePerRow[row])
					{
						minValuePerRow[row] = rawData[row, col];
					}

					if (rawData[row, col] > maxValuePerRow[row])
					{
						maxValuePerRow[row] = rawData[row, col];
					}
				}
			}

			for (int row = 0; row < rawData.GetLength(0); row++)
			{
				for (int col = 0; col<rawData.GetLength(1); col++)
				{
					scaledData[row, col] = (rawData[row, col] - minValuePerRow[row]) / (maxValuePerRow[row] - minValuePerRow[row]);
				}
			}
			return (scaledData, maxValuePerRow, minValuePerRow);
		}
	}
}
