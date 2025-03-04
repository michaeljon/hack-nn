using MathNet.Numerics.LinearAlgebra;

namespace Learning.Neural.Networks
{
    class Program
    {
        static void Main(string[] args)
        {
            // var zeros = ImageLoader.LoadImages("/Users/michaeljon/src/ml/mnist-png/training/0");

            var network = new NeuralNetwork([2, 3, 3, 1]);

            var costs = network.Train(
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

            // Console.WriteLine($"network cost first round {string.Join(", ", costs)}");
        }
    }
}
