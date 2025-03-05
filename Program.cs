using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Learning.Neural.Networks
{
    public class Program
    {
        private static readonly int epochs = 100;

        private static readonly double learningRate = 0.9;

        public static void Main(string[] _)
        {
            Console.WriteLine($"Loading training data --{DateTime.Now:s}--");
            var trainingSamples = ImageLoader.ReadTrainingData().ToList().Take(10000);

            Console.WriteLine($"Loading test data --{DateTime.Now:s}--");
            var testSamples = ImageLoader.ReadTestData().ToList().Take(10);

            var network = new NeuralNetwork(trainingSamples.ElementAt(0).Data.Length, 100, 20, 10);
            var layerName = string.Join("_", network.Layers.Select(n => $"[{n.NeuronCount}]"));

            var best = 0.0;

            Console.WriteLine($"Training model --{DateTime.Now:s}--");
            for (var epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var trainingSample in trainingSamples)
                {
                    network.ForwardPropagation(trainingSample.Data);
                    network.CalculateGradients(trainingSample.Targets);
                    network.UpdateParameters(learningRate);
                }

                var result = GetAccuracy(network, trainingSamples);

                Console.WriteLine();
                var outcome = best > 0 ?
                    result == best ? " (same)" :
                        result > best ? " (better)" : " (worse)" :
                    string.Empty;
                Console.WriteLine($"[{layerName}] Epoch #{epoch}, Accuracy=={result:P}{outcome} --{DateTime.Now:s}--");

                if (result > best)
                {
                    best = result;

                    Task.Run(() => NetworkSerializer.SaveNetwork(network, "saved", epoch, result, layerName));
                }
            }

            Console.WriteLine($"Testing model --{DateTime.Now:s}--");
        }

        private static double GetAccuracy(NeuralNetwork network, IEnumerable<Image> samples)
        {
            var matches = 0.0;

            foreach (var sample in samples)
            {
                var outputs = network.ForwardPropagation(sample.Data);
                var max = outputs.Max();
                var digit = outputs.ToList().IndexOf(max);

                matches += digit == sample.Label ? 1 : 0;
            }

            return matches / samples.Count();
        }
    }
}
