using System;
using System.Threading.Tasks;
using MessagePack;

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
            for (var i = 0; i < inputs.Length; i++)
            {
                Layers[0].Outputs[i] = inputs[i];
            }

            for (var l = 1; l < Layers.Length; l++)
            {
                var currentLayer = Layers[l];
                var previousLayer = Layers[l - 1];

                Parallel.For(0, currentLayer.NeuronCount, (j) =>
                {
                    var output = currentLayer.Biases[j];

                    for (var n = 0; n < previousLayer.NeuronCount; n++)
                    {
                        output += previousLayer.Outputs[n] * currentLayer.Weights[j][n];
                    }

                    currentLayer.Outputs[j] = Sigmoid(output);
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

                Parallel.For(0, currentLayer.NeuronCount, (i) =>
                {
                    var sum = 0.0;

                    for (var j = 0; j < nextLayer.NeuronCount; j++)
                    {
                        sum += nextLayer.Weights[j][i] * nextLayer.Gradients[j];
                    }

                    currentLayer.Gradients[i] = sum * SigmoidDerivative(currentLayer.Outputs[i]);
                });
            }
        }

        public void UpdateParameters(double learningRate)
        {
            for (var l = 1; l < Layers.Length; l++)
            {
                var currentLayer = Layers[l];
                var previousLayer = Layers[l - 1];

                Parallel.For(0, currentLayer.NeuronCount, (i) =>
                {
                    for (var j = 0; j < previousLayer.NeuronCount; j++)
                    {
                        currentLayer.Weights[i][j] -= learningRate * currentLayer.Gradients[i] * previousLayer.Outputs[j];
                        currentLayer.Biases[i] -= learningRate * currentLayer.Gradients[i];
                    }
                });
            }
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
