using System;
using MessagePack;

namespace Learning.Neural.Networks
{
    [MessagePackObject]
    public class Layer
    {
        private static readonly Random random = new();

        [Key(0)]
        public double[][] Weights { get; set; } = [];

        [Key(1)]
        public double[] Biases { get; set; }

        [Key(2)]
        public double[] Gradients { get; set; }

        [Key(3)]
        public double[] Outputs { get; set; }

        [Key(4)]
        public int NeuronCount { get; init; }

        [Key(5)]
        public int PrevLayerNeuronCount { get; init; }

        public Layer() { }

        public Layer(int thisLayerNeuronCount, int prevLayerNeuronCount)
        {
            NeuronCount = thisLayerNeuronCount;
            PrevLayerNeuronCount = prevLayerNeuronCount;

            Gradients = new double[NeuronCount];
            Outputs = new double[NeuronCount];
            Biases = new double[NeuronCount];
            Weights = new double[NeuronCount][];

            for (var n = 0; n < NeuronCount; n++)
            {
                // don't do the input layer (neuronCount == 0)
                if (prevLayerNeuronCount > 0)
                {
                    Weights[n] = new double[PrevLayerNeuronCount];

                    for (var w = 0; w < Weights[n].Length; w++)
                    {
                        Weights[n][w] = GetRandom();
                    }

                    Biases[n] = GetRandom() / Math.Sqrt(PrevLayerNeuronCount);
                }
            }
        }

        private static double GetRandom()
        {
            var x1 = 1 - random.NextDouble();
            var x2 = 1 - random.NextDouble();

            return Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
        }
    }
}
