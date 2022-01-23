namespace MGroup.MachineLearning.Tests
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.Drawing;
	using System.Linq;
	using System.Reflection;

	using MGroup.MachineLearning;
	using Microsoft.ML;
	using Microsoft.ML.Data;
	using Xunit;
	using Normalization;
	using System.IO;

	internal class FahrenheitDegrees
	{
		[ColumnName("input")]
		public float InputValue { get; set; }
	}

	internal class CelsiusPrediction
	{
		[ColumnName("output")]
		public float[] PredictedValue { get; set; }
	}

	public static class NeuralNetworkONNXtest
	{

		private static float testValue = 100;

		[Fact]
		private static void TestFtoCONNXmodel()
		{
			var context = new MLContext();

			var emptyData = new List<FahrenheitDegrees>();
			var data = context.Data.LoadFromEnumerable(emptyData);

			var pipeline = context.Transforms.ApplyOnnxModel(modelFile: "./ONNXmodels/convertFtoCtest.onnx", outputColumnName: "output", inputColumnName: "input");

			var model = pipeline.Fit(data);

			var predictionEngine = context.Model.CreatePredictionEngine<FahrenheitDegrees, CelsiusPrediction>(model);
	
			var prediction = predictionEngine.Predict(new FahrenheitDegrees { InputValue = testValue });

			double totalError = 0.0;
			Assert.True(totalError < 0.01);
		}
	}
}
