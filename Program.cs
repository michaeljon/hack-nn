using System;
using System.ComponentModel;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Neural.Networks
{
    class Program
    {
        static void Main(string[] args)
        {
            var network = new NeuralNetwork([784, 8, 8, 1]);

            var (X_train, y_train) = ImageLoader.LoadImagesAsMatrix("/Users/michaeljon/src/ml/mnist-png/training");
            RunImageNetwork(network, X_train, y_train);

            var (X_test, y_test) = ImageLoader.LoadImagesAsMatrix("/Users/michaeljon/src/ml/mnist-png/testing");
            for (var t = 0; t < X_test.RowCount; t++)
            {
                var test = X_test.Row(t).ToRowMatrix();
                var (yhat, rest) = network.FeedForward(test);

                if (yhat[0, 0] != y_test[t, 0])
                {
                    Console.WriteLine($"Test {t} expected {y_test[t, 0]:N0} but predicted {yhat[0, 0]:N0}");
                }
            }
        }

        static void RunImageNetwork(NeuralNetwork network, Matrix<double> X, Matrix<double> y)
        {
            var yhat = network.Train(
                X,
                y,
                alpha: 0.01,
                scale: true,
                epsilon: 0.0001,
                epochs: 1000
            );
        }

        static void RunExampleNetwork()
        {
            var network = new NeuralNetwork([2, 3, 3, 1]);

            var yhat = network.Train(
                Matrix<double>.Build.DenseOfArray(
                    new double[,]
                    {
                        { 150, 70 },    // it's our boy Jimmy again! 150 pounds, 70 inches tall.
                        { 254, 73 },
                        { 312, 68 },
                        { 120, 60 },
                        { 154, 61 },
                        { 212, 65 },
                        { 216, 67 },
                        { 145, 67 },
                        { 184, 64 },
                        { 130, 69 }
                    }
                ),
                Matrix<double>.Build.DenseOfArray(
                    new double[,]
                    {
                        { 0 },  // whew, thank God Jimmy isn't at risk for cardiovascular disease.
                        { 1 },  // damn, this guy wasn't as lucky
                        { 1 },  // ok, this guy should have seen it coming. 5"8, 312 lbs isn't great.
                        { 0 },
                        { 0 },
                        { 1 },
                        { 1 },
                        { 0 },
                        { 1 },
                        { 0 }
                    }
                ),
                alpha: 0.01,
                epsilon: 0.0001,
                epochs: 1000
            );
        }
    }
}
