using System;
using System.Linq;
using MathNet.Numerics;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;


namespace Learning.Neural.Networks
{
    internal class NeuralNetwork
    {
        private readonly IContinuousDistribution distribution = new Normal();

        private readonly int layerCount;
        private readonly int[] layerSizes;

        public NeuralNetwork(int[] layerSizes, bool randomWeights = true, bool randomBiases = true)
        {
            ArgumentNullException.ThrowIfNull(layerSizes);

            if (layerSizes.Length < 3)
            {
                throw new ArgumentOutOfRangeException(
                    "A network needs at least 3 layers. One input, one output, and at least one hidden.",
                    nameof(layerSizes));
            }

            this.layerSizes = layerSizes;
            layerCount = layerSizes.Length;

            Console.WriteLine("Feature count: {0}", layerSizes[0]);

            for (var l = 1; l < layerCount - 1; l++)
            {
                Console.WriteLine("Layer {0} size: {1}", l, this.layerSizes[l]);
            }
            Console.WriteLine("Output layer size: {0}", layerSizes[^1]);

            InitializeWeights(randomWeights);
            InitializeBiases(randomBiases);
        }

        public Matrix<double> Yhat { get; set; }

        public Matrix<double>[] Weights { get; set; }

        public Matrix<double>[] Biases { get; set; }

        private void InitializeWeights(bool randomWeights = true)
        {
            Weights = new Matrix<double>[layerCount];
            for (var w = 0; w < layerCount - 1; w++)
            {
                if (randomWeights == true)
                {
                    Weights[w + 1] = Matrix<double>.Build.Random(
                        layerSizes[w + 1],
                        layerSizes[w],
                        distribution);
                }
                else
                {
                    Weights[w + 1] = Matrix<double>.Build.Dense(
                        layerSizes[w + 1],
                        layerSizes[w],
                        1.0);
                }

                Console.WriteLine("Weights for layer {0} shape: {1}", w + 1, Weights[w + 1].ShapeAsString());
            }
        }

        private void InitializeBiases(bool randomBiases = true)
        {
            Biases = new Matrix<double>[layerCount];
            for (var b = 1; b < layerCount; b++)
            {
                if (randomBiases == true)
                {
                    Biases[b] = Matrix<double>.Build.Random(
                        layerSizes[b],
                        1,
                        distribution);
                }
                else
                {
                    Biases[b] = Matrix<double>.Build.Dense(
                        layerSizes[b],
                        1,
                        1.0);
                }

                Console.WriteLine("Biases for layer {0} shape: {1}", b, Biases[b].ShapeAsString());
            }
        }

        public (double[] cposts, int epochs) Train(Matrix<double> trainingData, Matrix<double> labels, double alpha, bool scale = true, double epsilon = 0.001, int epochs = 1000)
        {
            var costs = new double[epochs + 1];
            var y = labels.Transpose();
            int m = trainingData.RowCount;

            if (scale == true)
            {
                Console.WriteLine("Scaling training data");
                Scale(trainingData);
            }

            Matrix<double> yhat = default;
            Matrix<double>[] rest = default;

            Console.WriteLine("Training model");
            var epoch = 0;
            while (epoch++ < epochs)
            {
                (yhat, rest) = FeedForward(trainingData);

                var error = ComputeCost(yhat, y);
                costs[epoch - 1] = error;

                Matrix<double> propagator = null;
                Matrix<double>[] dC_dWs = new Matrix<double>[layerCount];
                Matrix<double>[] dC_dbs = new Matrix<double>[layerCount];

                for (var l = layerCount - 1; l > 0; l--)
                {
                    var (dC_dW, dC_db, dC_dA) =
                        BackProp(yhat, y, propagator, m, l, anext: rest[l], aprev: rest[l - 1], w: Weights[l]);

                    dC_dWs[l] = dC_dW;
                    dC_dbs[l] = dC_db;

                    propagator = dC_dA;
                }

                for (var l = layerCount - 1; l > 0; l--)
                {
                    Weights[l] = Weights[l] - (alpha * dC_dWs[l]);
                    Biases[l] = Biases[l] - (alpha * dC_dbs[l]);
                }

                if (epoch % 20 == 0)
                {
                    // Console.WriteLine($"epoch {epoch}: cost = {cost:F4}");
                }

                if (epsilon != 0)
                {
                    if (Math.Abs(costs[epoch] - costs[epoch - 1]) < epsilon)
                    {
                        break;
                    }
                }
            }

            Yhat = yhat;

            return (costs[1..], epoch);
        }

        public (Matrix<double> yhat, Matrix<double>[] rest) FeedForward(Matrix<double> X)
        {
            var a = new Matrix<double>[layerCount];
            var m = X.RowCount;

            a[0] = X.Transpose();
            // Console.WriteLine("A0 shape: {0}", a[0].ShapeAsString());

            for (var l = 1; l < layerCount; l++)
            {
                var z = Weights[l] * a[l - 1] + Biases[l].Broadcast(m);
                z.AssertShape(layerSizes[l], m);
                a[l] = z.Map(SpecialFunctions.Logistic);

                // Console.WriteLine("A{0} shape: {1}", l, a[l].ShapeAsString());
            }

            return (a[^1], a);
        }

        private (Matrix<double> dC_dW, Matrix<double> dC_db, Matrix<double> dC_da) BackProp(Matrix<double> yhat, Matrix<double> y, Matrix<double> propagator, int m, int l, Matrix<double> anext, Matrix<double> aprev, Matrix<double> w)
        {
            Matrix<double> dC_dZ;

            if (l == layerCount - 1)
            {
                // step 1. calculate dC/dZ[l] using shorthand we derived earlier
                dC_dZ = (1.0 / (double)m) * (yhat - y);
            }
            else
            {
                // step 1. calculate dC/dZ[l] = dC/dA[l] * dA[l]/dZ[l]
                var dA_dZ = anext.PointwiseMultiply(1 - anext);
                dC_dZ = propagator.PointwiseMultiply(dA_dZ);
            }
            dC_dZ.AssertShape(layerSizes[l], m);

            var dZ_dW = aprev;
            dZ_dW.AssertShape(layerSizes[l - 1], m);

            var dC_dW = dC_dZ * dZ_dW.Transpose();
            dC_dW.AssertShape(layerSizes[l], layerSizes[l - 1]);

            var dC_db = dC_dZ.SumAcrossRows();
            dC_db.AssertShape(layerSizes[l], 1);

            if (l > 1)
            {
                var dC_dA = w.Transpose() * dC_dZ;
                dC_dA.AssertShape(layerSizes[l - 1], m);

                return (dC_dW, dC_db, dC_dA);
            }
            else
            {
                return (dC_dW, dC_db, null);
            }
        }

        private static double ComputeCost(Matrix<double> yhat, Matrix<double> y)
        {
            var ys = yhat.Row(0).AsEnumerable();
            var yhs = y.Row(0).AsEnumerable();

            var cs = ys.Zip(yhs)
                .Select((pair, _) =>
                {
                    var yhi = pair.First;
                    var yi = pair.Second;

                    return -(yi * Math.Log(yhi) + (1.0 - yi) * Math.Log(1.0 - yhi));
                });

            return cs.Sum() / cs.Count();
        }

        private static void Scale(Matrix<double> matrix)
        {
            for (var col = 0; col < matrix.ColumnCount; col++)
            {
                var column = matrix.Column(col).ToArray();

                var mean = column.Mean();
                var stdDev = Statistics.StandardDeviation(column);

                var scaled = column.Select(x => (stdDev != 0) ? (x - mean) / stdDev : 0).ToArray();

                for (int i = 0; i < matrix.RowCount; i++)
                {
                    matrix[i, col] = scaled[i];
                }
            }
        }
    }
}
