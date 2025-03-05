using System;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace Learning.Neural.Networks
{
    class Program
    {
        static void Main(string[] args)
        {
            RunImageNetworkAgain();
        }

        static void RunXorNetwork()
        {
            var network = new NeuralNetwork([3, 8, 1]);

            var X_train = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 0, 0, 1 },
                    { 1, 1, 1 },
                    { 1, 0, 1 },
                    { 0, 1, 1 },
                }
            );

            var y_train = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 0 },
                    { 1 },
                    { 1 },
                    { 0 },
                }
            );

            var (costs, epochs) = network.Train(
                X_train,
                y_train,
                alpha: 0.001,
                scale: false,
                epsilon: 0.00001,
                epochs: 1);

            Console.WriteLine("Weights");
            network.Weights.Skip(1).Each((m, i) => { Console.WriteLine($"W{i + 1}"); m.Print(); });

            Console.WriteLine("Biases");
            network.Biases.Skip(1).Each((m, i) => { Console.WriteLine($"b{i + 1}"); m.Print(); });
        }

        static void RunImageNetwork()
        {
            var network = new NeuralNetwork([784, 256, 256, 10, 1]);

            var (X_train, y_train) = ImageLoader.LoadImagesAsMatrix("/Users/michaeljon/src/ml/mnist-png/training", count: 100);
            network.Train(
                X_train,
                y_train,
                alpha: 0.01,
                scale: false,
                epsilon: 0.0,
                epochs: 100
            );

            Console.WriteLine("Ŷ");
            network.Yhat.Print();

            var (X_test, y_test) = ImageLoader.LoadImagesAsMatrix("/Users/michaeljon/src/ml/mnist-png/testing", count: 100);
            for (var t = 0; t < X_test.RowCount; t++)
            {
                var test = X_test.Row(t).ToRowMatrix();
                var (yhat, rest) = network.FeedForward(test);

                if (yhat[0, 0] != y_test[t, 0])
                {
                    Console.WriteLine($"Test {t} expected {y_test[t, 0]:N0} but predicted {yhat[0, 0]:N0}");
                }
            }

            // Console.WriteLine("Weights");
            // network.Weights[1].Print();
            // network.Weights.Skip(1).Each((m, i) => { Console.WriteLine($"W{i + 1}"); m.Print(); });

            // Console.WriteLine("Biases");
            // network.Biases.Skip(1).Each((m, i) => { Console.WriteLine($"b{i + 1}"); m.Print(); });
        }

        static void RunExampleNetwork()
        {
            var network = new NeuralNetwork([2, 3, 3, 1], false, false);

            var (costs, epochs) = network.Train(
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
                scale: false,
                epsilon: 0.0,
                epochs: 1000
            );

            var (yhat, rest) = network.FeedForward(
                Matrix<double>.Build.DenseOfArray(
                    new double[,]
                    {
                        { 150, 70 },    // it's our boy Jimmy again! 150 pounds, 70 inches tall.
                    }
                ));

            // Console.WriteLine("Costs");
            // Console.WriteLine($"[ {string.Join(", ", costs.Take(epochs))} ]");

            Console.WriteLine("Ŷ");
            network.Yhat.Print();

            Console.WriteLine("Weights");
            network.Weights.Skip(1).Each((m, i) => { Console.WriteLine($"W{i + 1}"); m.Print(); });

            Console.WriteLine("Biases");
            network.Biases.Skip(1).Each((m, i) => { Console.WriteLine($"b{i + 1}"); m.Print(); });
        }

        static void RunImageNetworkAgain()
        {
            var network = new NeuralNetwork([784, 256, 256, 10, 1]);

            var rawData = RawImageLoader.ReadTrainingData().Take(100);

            var X_train = Matrix<double>.Build.DenseOfRows(rawData.Select(i => i.Data));
            var y_train = Vector<double>.Build.DenseOfArray([.. rawData.Select(i => i.Label)]).ToColumnMatrix();

            network.Train(
                X_train,
                y_train,
                alpha: 0.01,
                scale: false,
                epsilon: 0.0,
                epochs: 100
            );

            Console.WriteLine("Ŷ");
            network.Yhat.Print();

            foreach (var image in RawImageLoader.ReadTestData().Take(10))
            {
                var X_test = Vector<double>.Build.DenseOfArray(image.Data).ToRowMatrix();
                var y_test = Matrix<double>.Build.DenseOfArray(new double[,] { { image.Label } });

                var (yhat, rest) = network.FeedForward(X_test);

                if (yhat[0, 0] != y_test[0, 0])
                {
                    Console.WriteLine($"Test expected {y_test[0, 0]:N0} but predicted {yhat[0, 0]:N0}");
                }
            }


            // Console.WriteLine("Weights");
            // network.Weights[1].Print();
            // network.Weights.Skip(1).Each((m, i) => { Console.WriteLine($"W{i + 1}"); m.Print(); });

            // Console.WriteLine("Biases");
            // network.Biases.Skip(1).Each((m, i) => { Console.WriteLine($"b{i + 1}"); m.Print(); });
        }
    }
}
