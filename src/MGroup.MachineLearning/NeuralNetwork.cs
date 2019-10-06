namespace MGroup.MachineLearning
{
	using System;
	using System.Linq;
	using System.Collections.Generic;

	using NumSharp;
	using Tensorflow;

	using static Tensorflow.Binding;
	using MGroup.MachineLearning.Normalization;

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
		/// This class constucts a neural network that best 
		/// approximates the relationship between a dependent 
		/// variable and some independent ones.
		/// <summary>
		public NeuralNetwork(int numHiddenLayers, int trainingEpochs, INormalization normalization)
		{
			this.normalization = normalization;
			this.trainingEpochs = trainingEpochs;
			this.numHiddenLayers = numHiddenLayers;
		}

		public bool IsImportingGraph { get; set; } = false;

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

			// Shape [4]
			var predictor = tf.tanh(tf.squeeze(logits));
			var loss = tf.reduce_mean(tf.square(predictor - tf.cast(labels, tf.float32)), name: "loss");

			var gs = tf.Variable(0, trainable: false, name: "global_step");
			var train_op = tf.train.GradientDescentOptimizer(0.2f).minimize(loss, global_step: gs);

			return (train_op, loss, gs, predictor);
		}

		/// <summary>
		/// Evaluate the weight coefficients and biases so that
		/// the mean-square error between the between the 
		/// known values of the dependent variable and the 
		/// ones predicted
		/// <summary>
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
		/// Utilize the neural network
		/// to make new predictions
		/// </summary>
		public float[] Predict(double[,] X)
		{
			var output = sess.run(predictor, new FeedItem(features, np.array(X)));
			return output.ToArray<float>();

		}

		/// <summary>
		/// test the accuracy of the neural network
		/// on additional test data
		/// </summary>
		public void Test(Session sess)
		{
			throw new NotImplementedException();
		}

		public Graph BuildGraph()
		{
			throw new NotImplementedException();
		}

	}
}
