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

        private Matrix<double>[] weights;
        private Matrix<double>[] biases;

        public NeuralNetwork(int[] layerSizes)
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

            InitializeWeights();
            InitializeBiases();
        }

        private void InitializeWeights()
        {
            weights = new Matrix<double>[layerCount];
            for (var w = 0; w < layerCount - 1; w++)
            {
                weights[w + 1] = Matrix<double>.Build.Random(
                    layerSizes[w + 1],
                    layerSizes[w],
                    distribution);

                Console.WriteLine("Weights for layer {0} shape: {1}", w + 1, weights[w + 1].ShapeAsString());
            }
        }

        private void InitializeBiases()
        {
            biases = new Matrix<double>[layerCount];
            for (var b = 1; b < layerCount; b++)
            {
                biases[b] = Matrix<double>.Build.Random(
                    layerSizes[b],
                    1,
                    distribution);

                Console.WriteLine("Biases for layer {0} shape: {1}", b, biases[b].ShapeAsString());
            }
        }

        public double[] Train(Matrix<double> trainingData, Matrix<double> labels, double alpha, bool scale = true, double epsilon = 0.001, int epochs = 1000)
        {
            var costs = new double[epochs + 1];
            var y = labels.Transpose();
            int m = trainingData.RowCount;

            if (scale == true)
            {
                Console.WriteLine("Scaling training data");

                Scale(trainingData);
            }

            for (var epoch = 1; epoch <= epochs; epoch++)
            {
                var (yhat, rest) = FeedForward(trainingData);

                var cost = ComputeCost(yhat, y);
                costs[epoch] = cost;

                Matrix<double> propagator = null;

                for (var l = layerCount - 1; l > 0; l--)
                {
                    var (dC_dW, dC_db, dC_dA) = BackProp(yhat, y, propagator, m, l, anext: rest[l], aprev: rest[l - 1], w: weights[l]);

                    weights[l] = weights[l] - (alpha * dC_dW);
                    biases[l] = biases[l] - (alpha * dC_db);

                    propagator = dC_dA;
                }

                if (epoch % 20 == 0)
                {
                    Console.WriteLine($"epoch {epoch}: cost = {cost:F4}");
                }

                if (Math.Abs(costs[epoch] - costs[epoch - 1]) < epsilon)
                {
                    break;
                }
            }

            return costs[1..];
        }

        private (Matrix<double> yhat, Matrix<double>[] rest) FeedForward(Matrix<double> trainingData)
        {
            var a = new Matrix<double>[layerCount];
            var m = trainingData.RowCount;

            a[0] = trainingData.Transpose();
            // Console.WriteLine("A0 shape: {0}", a[0].ShapeAsString());

            for (var l = 1; l < layerCount; l++)
            {
                var z = weights[l] * a[l - 1] + biases[l].Broadcast(m);
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

                var scaled = column.Select(x => (x - mean) / stdDev).ToArray();

                for (int i = 0; i < matrix.RowCount; i++)
                {
                    matrix[i, col] = scaled[i];
                }
            }
        }
    }
}
