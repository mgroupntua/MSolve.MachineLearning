namespace MGroup.MachineLearning
{
	using System;
	using System.Linq;
	using System.Collections.Generic;

	using NumSharp;
	using Tensorflow;

	using static Tensorflow.Binding;

	/// <summary>
	/// This class provides the optimal linear regression
	/// coefficients for a given training data set, tests
	/// the accuracy of the approximation using a testing
	/// data set and predicts the value of the dependent
	/// variable on new values of the independent variables
	/// <summary>
	public class LinearRegression : ILinearRegression
	{
		private readonly int trainingEpochs;
		private readonly float learningRate;

		Tensor predictor;
		Tensor features;
		Tensor weights;
		Tensor bias;
		Session sess;

		public LinearRegression(int trainingEpochs, float learningRate)
		{
			this.trainingEpochs = trainingEpochs;
			this.learningRate = learningRate;
		}

		/// <summary>
		/// Evaluate the regression coefficient using a gradient descent
		/// algorithm that minimizes the least squares error between the 
		/// known values of the dependent variable and the ones predicted
		/// </summary>
		public void Train(double[] X, double[] Y)
		{
			// tf Graph Input
			features = tf.placeholder(tf.float32);
			var labels = tf.placeholder(tf.float32);
			sess = new Session();

			// Set model weights 
			// We can set a fixed init value in order to debug
			// var rnd1 = rng.randn<float>();
			// var rnd2 = rng.randn<float>();
			weights = tf.Variable(-0.06f, name: "weight");
			bias = tf.Variable(-0.73f, name: "bias");

			// Construct a linear model
			var pred = tf.add(tf.multiply(features, weights), bias);
			this.predictor = pred;
			// Mean squared error
			var cost = tf.reduce_sum(tf.pow(pred - labels, 2.0f)) / (2.0f * X.GetLength(0));

			// Gradient descent
			// Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
			var optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost);

			// Initialize the variables (i.e. assign their default value)
			var init = tf.global_variables_initializer();

			// Start training
			// Run the initializer
			sess.run(init);

				// Fit all training data
			for (int epoch = 0; epoch < trainingEpochs; epoch++)
				{
					foreach (var (x, y) in zip<float>(np.array(X), np.array(Y)))
						sess.run(optimizer, (features, x), (labels, y));
				}

			var training_cost = sess.run(cost, (features, np.array(X)), (labels, np.array(Y)));
		}

		/// <summary>
		/// Utilize the calculated regression model
		/// to make new predictions
		/// </summary>
		public (float[], float, float) Predict(double[] X)
		{
			var output = sess.run(predictor, new FeedItem(features, np.array(X)));
			var optimalWeights = sess.run(weights);
			var optimaldBias = sess.run(bias);
			return (output.ToArray<float>(), optimalWeights, optimaldBias);
		}

		/// <summary>
		/// test the accuracy of the regression model
		/// on additional test data
		/// </summary>
		public void Test(Session sess)
		{
			throw new NotImplementedException();
		}
	}
}
