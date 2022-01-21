namespace MGroup.MachineLearning.Tests
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.Drawing;
	using System.Linq;
	using System.Reflection;

	using MGroup.MachineLearning;
	using MGroup.MachineLearning.UtilityFunctions;
	using MGroup.MachineLearning.DimensionalityReduction;
	using Xunit;

	public class DMAPexample
	{
		// DMAP algorithm parameters
		private int numberOfKNN = 512;
		private int NNofKDE = 64;
		private int differentialOperator = 1;  // 1:Laplace-Beltrami, 2:generator of grad system
		private int numberOfEigenvectors = 11;
		private double[,] dataSet = ImportData.ImportDataFromCSV();


		[Fact]
		private void TestDMAPAlgorithm()
		{
			DiffusionMapsAlgorithm DMAP = new DiffusionMapsAlgorithm(dataSet, numberOfKNN, NNofKDE, differentialOperator, numberOfEigenvectors);
			DMAP.ProcessData();
			Assert.True(Math.Abs(DMAP.DMAPeigenvalues[1] + 7.1048) < 0.001);
			Assert.True(Math.Abs(DMAP.DMAPeigenvalues[2] + 12.8862) < 0.001);
			Assert.True(Math.Abs(DMAP.DMAPeigenvalues[3] + 16.8511) < 0.001);
		}
	}
}
