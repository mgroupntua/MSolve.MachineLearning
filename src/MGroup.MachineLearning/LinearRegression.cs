namespace MGroup.MachineLearning
{
	using System;
	using System.Linq;
	using System.Collections.Generic;

	using NumSharp;
	using Tensorflow;

	using static Tensorflow.Binding;

	/// <summary>
	/// This class constucts a a linear regression model that best approximates the
	/// relationship between a dependent variable and a set of independent variables.
	/// </summary>
	public class LinearRegression : ILinearRegression
	{
		private readonly int trainingEpochs;
		private readonly float learningRate;

		Tensor predictor;
		Tensor features;
		Tensor weights;
		Tensor bias;
		Session sess;

		/// <summary>
		/// Assigns values to the parameters needed for the implementation of the regression model.
		/// </summary>
		/// <param name="trainingEpochs">An <see cref="int"/> corresponding to the number of times of the training vectors are used once
		/// to update the weight and bias.</param>
		/// <param name="learningRate">A <see cref="float"/> used for tuning the gradient descent optimizer.</param>
		public LinearRegression(int trainingEpochs, float learningRate)
		{
			this.trainingEpochs = trainingEpochs;
			this.learningRate = learningRate;
		}

		/// <summary>
		/// Evaluates the optimal coefficients of the regression that minimize the least squares error
		/// between the given values of the dependent variables and the ones predicted.
		/// </summary>
		/// <param name="X"> A <see cref="double"/> array containing the given values of the independent variables.</param>
		/// <param name="Y"> A <see cref="double"/> array containing the corresponding values of the dependent variable.</param>
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
		/// Utilizes the regression model to make predictions for new data.
		/// </summary>
		/// <param name="X">A <see cref="double"/> array containing new values of the independent variables,
		/// whose outcome we want to predict.</param>
		/// <returns> A <see cref="float"/> array containing the predicted values of the dependent variables, a <see cref="float"/>
		/// containing the optimal weight and a <see cref="float"/> containing the optimal bias for the model.</returns>
		public (float[], float, float) Predict(double[] X)
		{
			var output = sess.run(predictor, new FeedItem(features, np.array(X)));
			var optimalWeights = sess.run(weights);
			var optimaldBias = sess.run(bias);
			return (output.ToArray<float>(), optimalWeights, optimaldBias);
		}

		/// <summary>
		/// tests the accuracy of the regression model on additional test data.
		/// </summary>
		/// <param name="X"> A <see cref="double"/> array containing the additional values of the independent variables.</param>
		/// <param name="Y"> A <see cref="double"/> array containing the corresponding values of the dependent variable.</param>
		public void Test(double[] X, double[] Y)
		{
			throw new NotImplementedException();
		}
	}
}
