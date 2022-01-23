namespace MGroup.MachineLearning.Tests
{
	using System;
	using System.Collections.Generic;
	using System.IO;
	using System.Text;
	using System.Linq;

	using MGroup.MachineLearning;
	using Xunit;
	using Microsoft.ML;

	public static class IrisClusteringExample
	{
		// data input
		private static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "iris.data");

		private static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "IrisClusteringModel.zip");

		[Fact]
		private static void TestIrisClustering()
		{
			var mlContext = new MLContext(seed: 0);

			IDataView dataView = mlContext.Data.LoadFromTextFile<DeepLearning.IrisData>(_dataPath, hasHeader: false, separatorChar: ',');

			string featuresColumnName = "Features";

			var pipeline = mlContext.Transforms.Concatenate(featuresColumnName, "SepalLength","SepalWidth","PetalLength","PetalWidth")
				.Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));

			var model = pipeline.Fit(dataView);

			using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
			{
				mlContext.Model.Save(model, dataView.Schema, fileStream);
			}

			var predictor = mlContext.Model.CreatePredictionEngine<DeepLearning.IrisData, DeepLearning.ClusterPrediction>(model);

			var prediction = predictor.Predict(DeepLearning.TestIrisData.Setosa);

			Assert.True(prediction.PredictedClusterID == 1);

		}
	}
}
