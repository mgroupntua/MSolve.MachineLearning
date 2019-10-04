namespace MGroup.MachineLearning
{
	using System;
	using System.Linq;
	using System.Collections.Generic;

	using NumSharp;
	using Tensorflow;

	using static Tensorflow.Binding;

	public class LinearRegression : ILinearRegression
	{
		public double[] InputX { get; set; }

		public double[] InputY { get; set; }

		public float LearningRate { get; set; }

		public int TrainingEpochs { get; set; }

		public int DisplayStep { get; set; }

		int nSamples;

		NDArray trainX;
		NDArray trainY;

		public float Weight { get; set; }

		public float Bias { get; set; }

		public bool IsImportingGraph { get; set; } = false;

		public void Train()
		{
			// Training Data
			PrepareData(InputX, InputY);

			// tf Graph Input
			var X = tf.placeholder(tf.float32);
			var Y = tf.placeholder(tf.float32);

			// Set model weights 
			// We can set a fixed init value in order to debug
			// var rnd1 = rng.randn<float>();
			// var rnd2 = rng.randn<float>();
			var W = tf.Variable(-0.06f, name: "weight");
			var b = tf.Variable(-0.73f, name: "bias");

			// Construct a linear model
			var pred = tf.add(tf.multiply(X, W), b);

			// Mean squared error
			var cost = tf.reduce_sum(tf.pow(pred - Y, 2.0f)) / (2.0f * nSamples);

			// Gradient descent
			// Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
			var optimizer = tf.train.GradientDescentOptimizer(LearningRate).minimize(cost);

			// Initialize the variables (i.e. assign their default value)
			var init = tf.global_variables_initializer();

			// Start training
			using (var sess = tf.Session())
			{
				// Run the initializer
				sess.run(init);

				// Fit all training data
				for (int epoch = 0; epoch < TrainingEpochs; epoch++)
				{
					foreach (var (x, y) in zip<float>(trainX, trainY))
						sess.run(optimizer, (X, x), (Y, y));

					// Display logs per epoch step
					if ((epoch + 1) % DisplayStep == 0)
					{
						var c = sess.run(cost, (X, trainX), (Y, trainY));
						Console.WriteLine($"Epoch: {epoch + 1} cost={c} " + $"W={sess.run(W)} b={sess.run(b)}");
					}
				}

				var training_cost = sess.run(cost, (X, trainX), (Y, trainY));

				Weight = sess.run(W);
				Bias = sess.run(b);
			}
		}

		public void PrepareData(double[] inputX, double[] inputY)
		{
			trainX = np.array(inputX);
			trainY = np.array(inputY);
			nSamples = trainX.shape[0];
		}

		public Graph ImportGraph()
		{
			throw new NotImplementedException();
		}

		public Graph BuildGraph()
		{
			throw new NotImplementedException();
		}

		public void Predict(Session sess)
		{
			throw new NotImplementedException();
		}

		public void Test(Session sess)
		{
			throw new NotImplementedException();
		}
	}
}
