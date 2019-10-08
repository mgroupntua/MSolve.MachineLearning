namespace MGroup.MachineLearning
{
	using System;
	using System.Linq;
	using System.Collections.Generic;

	using NumSharp;
	using Tensorflow;

	using static Tensorflow.Binding;
	using MGroup.MachineLearning.Normalization;

	/// <summary>
	/// This class constucts a neural network that best
	/// approximates the relationship between a dependent
	/// variable and a set of independent variables.
	/// </summary>
	public class NeuralNetwork : INeuralNetwork
	{
		private readonly INormalization normalization;
		private readonly int trainingEpochs;
		private readonly int numHiddenLayers;

		Graph graph;
		Tensor features;
		Tensor predictor;
		Session sess;

		/// <summary>
		/// Assigns values to the parameters needed for the implementation of the neural network.
		/// </summary>
		/// <param name="numHiddenLayers">An <see cref="int"/> corresponding to the no. of hidden layers in the network.</param>
		/// <param name="trainingEpochs">An <see cref="int"/> corresponding to the number of times all of the training vectors are used once
		/// to update the weights.</param>
		/// <param name="normalization">An <see cref="INormalization"/> refering to the method of choice to normalize the data.</param>
		public NeuralNetwork(int numHiddenLayers, int trainingEpochs, INormalization normalization)
		{
			this.normalization = normalization;
			this.trainingEpochs = trainingEpochs;
			this.numHiddenLayers = numHiddenLayers;
		}

		/// <summary>
		/// Evaluates the weight coefficients and biases of the neural network that minimize the mean-square error
		/// between the given values of the dependent variables and the ones predicted.
		/// </summary>
		/// <param name="X">A <see cref="double"/> array containing the given values of the independent variables.</param>
		/// <param name="Y">A <see cref="double"/> array containing the corresponding values of the dependent variable.</param>
		public void Train(double[,] X, float[] Y)
		{
			graph = tf.Graph().as_default();
			features = tf.placeholder(tf.float32, new TensorShape(X.GetLength(0), X.GetLength(1)));
			var labels = tf.placeholder(tf.int32, new TensorShape(X.GetLength(0)));
			(Operation train_op, Tensor loss, Tensor gs, Tensor predictions) = MakeGraph(features, labels, numHiddenLayers);
			this.predictor = predictions;
			var init = tf.global_variables_initializer();

			float loss_value = 0;
			sess = new Session();
			sess = tf.Session(graph);
			sess.run(init);
			var step = 0;

			while (step < trainingEpochs)
			{
				(_, step, loss_value) = sess.run((train_op, gs, loss), (features, np.array(X)), (labels, np.array(Y)));
			}
		}

		/// <summary>
		/// Utilize the neural network to make predictions for new data.
		/// </summary>
		/// <param name="X">A <see cref="double"/> array containing the new values of the independent variables, whose outcome we want to predict.</param>
		/// <returns> A <see cref="float"/> array containing the predicted values of the dependent variables.</returns>
		public float[] Predict(double[,] X)
		{
			var output = sess.run(predictor, new FeedItem(features, np.array(X)));
			return output.ToArray<float>();

		}

		/// <summary>
		/// Tests the accuracy of the neural network on additional data.
		/// </summary>
		/// <param name="X"> A <see cref="double"/> array containing the additional values of the independent variables.</param>
		/// <param name="Y"> A <see cref="double"/> array containing the corresponding values of the dependent variable.</param>
		public void Test(double[,] X, double[] Y)
		{
			throw new NotImplementedException();
		}

		private (Operation, Tensor, Tensor, Tensor) MakeGraph(Tensor features, Tensor labels, int numHiddenLayers)
		{
			var stddev = 1 / Math.Sqrt(2);
			var hidden_weights = tf.Variable(tf.truncated_normal(new int[] { 2, numHiddenLayers }, seed: 1, stddev: (float)stddev));

			var hidden_activations = tf.nn.relu(tf.matmul(features, hidden_weights));

			var output_weights = tf.Variable(tf.truncated_normal(
				new[] { numHiddenLayers, 1 },
				seed: 17,
				stddev: (float)(1 / Math.Sqrt(numHiddenLayers))
			));

			var logits = tf.matmul(hidden_activations, output_weights);

			var predictor = tf.tanh(tf.squeeze(logits));
			var loss = tf.reduce_mean(tf.square(predictor - tf.cast(labels, tf.float32)), name: "loss");

			var gs = tf.Variable(0, trainable: false, name: "global_step");
			var train_op = tf.train.GradientDescentOptimizer(0.2f).minimize(loss, global_step: gs);

			return (train_op, loss, gs, predictor);
		}

	}
}
