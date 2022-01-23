namespace MGroup.MachineLearning.DeepLearning
{
	using System;
	using System.Collections.Generic;
	using System.Text;

	using Microsoft.ML.Data;

	public class ClusterPrediction
	{
		[ColumnName("PredictedLabel")]
		public uint PredictedClusterID;

		[ColumnName("Score")]
		public float[] Distances;

	}
}
