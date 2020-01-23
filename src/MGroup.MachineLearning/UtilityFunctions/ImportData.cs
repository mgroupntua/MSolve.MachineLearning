using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;


namespace MGroup.MachineLearning.UtilityFunctions
{
	public class ImportData
	{
		public static double[,] ImportDataFromCSV()
		{
			var filePath = @"E:\GIANNIS_DATA\DESKTOP\VS\DMAPTestData.csv";
			string[][] dataValues = File.ReadLines(filePath).Select(x => x.Split(',')).ToArray();
			double[,] dataSet = new double[dataValues.GetLength(0), dataValues[0].Length];

			for (int i = 0; i < dataValues.GetLength(0); i++)
			{
				for (int j = 0; j < dataValues[i].Length; j++)
				{
					dataSet[i, j] = Convert.ToDouble(dataValues[i][j]);
				}
			}

			return dataSet;
		}
	}
}
