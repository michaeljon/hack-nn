using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using MessagePack;

// Some notes about the loops here
//
//  For the inner loops over adjacent layers, this class partitions the "neuron" count
//  automatically by the number of CPUs and number of items in the range. This partition
//  is then used as the target for a Parallel.ForEach().
//
//  This was done because a simple Parallel.For() would simply scatter all the work
//  across all the CPUs. That's bad because the inner loop is CPU intensive, but extremely
//  short. This leads to heavy context switching between the individual items in the
//  arrays.
//
//  Chunking this gives a balance between CPU-heavy processing (lots of neurons per CPU-task)
//  and context switching between the threads.
namespace Learning.Neural.Networks
{
    [MessagePackObject]
    public class NeuralNetwork
    {
        [Key(0)]
        public Layer[] Layers { get; set; } = [];

        public NeuralNetwork() { }

        public NeuralNetwork(params int[] layerSizes)
        {
            Layers = new Layer[layerSizes.Length];
            Layers[0] = new Layer(layerSizes[0], 0);

            for (var l = 1; l < layerSizes.Length; l++)
            {
                Layers[l] = new Layer(layerSizes[l], layerSizes[l - 1]);
            }
        }

        public double[] ForwardPropagation(double[] inputs)
        {
            Array.Copy(inputs, Layers[0].Outputs, inputs.Length);

            for (var l = 1; l < Layers.Length; l++)
            {
                var currentLayer = Layers[l];
                var previousLayer = Layers[l - 1];

                Parallel.ForEach(Partitioner.Create(0, currentLayer.NeuronCount), (range) =>
                {
                    for (var j = range.Item1; j < range.Item2; j++)
                    {
                        var output = currentLayer.Biases[j];

                        for (var n = 0; n < previousLayer.NeuronCount; n++)
                        {
                            output += previousLayer.Outputs[n] * currentLayer.Weights[j][n];
                        }

                        currentLayer.Outputs[j] = Sigmoid(output);
                    }
                });
            }

            return [.. Layers[^1].Outputs];
        }

        public void CalculateGradients(double[] targets)
        {
            // final layer
            for (var i = 0; i < Layers[^1].NeuronCount; i++)
            {
                Layers[^1].Gradients[i] = (Layers[^1].Outputs[i] - targets[i]) * SigmoidDerivative(Layers[^1].Outputs[i]);
            }

            // other layers, working backwards
            for (var l = Layers.Length - 2; l >= 1; l--)
            {
                var currentLayer = Layers[l];
                var nextLayer = Layers[l + 1];

                Parallel.ForEach(Partitioner.Create(0, currentLayer.NeuronCount), (range) =>
                {
                    for (var i = range.Item1; i < range.Item2; i++)
                    {
                        var sum = 0.0;

                        for (var j = 0; j < nextLayer.NeuronCount; j++)
                        {
                            sum += nextLayer.Weights[j][i] * nextLayer.Gradients[j];
                        }

                        currentLayer.Gradients[i] = sum * SigmoidDerivative(currentLayer.Outputs[i]);
                    }

                });
            }
        }

        public void UpdateParameters(double learningRate)
        {
            for (var l = 1; l < Layers.Length; l++)
            {
                var currentLayer = Layers[l];
                var previousLayer = Layers[l - 1];

                Parallel.ForEach(Partitioner.Create(0, currentLayer.NeuronCount), (range) =>
                {
                    for (var i = range.Item1; i < range.Item2; i++)
                    {
                        for (var j = 0; j < previousLayer.NeuronCount; j++)
                        {
                            currentLayer.Weights[i][j] -= learningRate * currentLayer.Gradients[i] * previousLayer.Outputs[j];
                        }

                        currentLayer.Biases[i] -= learningRate * currentLayer.Gradients[i];
                    }
                });
            }
        }

        public NeuralNetwork Clone()
        {
            var layers = new Layer[Layers.Length];

            for (var layer = 0; layer < Layers.Length; layer++)
            {
                layers[layer] = Layers[layer].Clone();
            }

            return new NeuralNetwork()
            {
                Layers = layers
            };
        }

        // sigmoid activation
        private static double Sigmoid(double weightedInput)
        {
            return 1.0 / (1.0 + Math.Exp(-weightedInput));
        }

        private static double SigmoidDerivative(double value)
        {
            return value * (1 - value);
        }
    }
}
