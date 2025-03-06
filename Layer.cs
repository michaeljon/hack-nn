using System;
using MessagePack;

namespace Learning.Neural.Networks
{
    [MessagePackObject]
    public partial class Layer
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
        public int NeuronCount { get; private set; }

        [Key(5)]
        public int PrevLayerNeuronCount { get; private set; }

        public Layer() { }

        public Layer(int thisLayerNeuronCount, int prevLayerNeuronCount)
        {
            NeuronCount = thisLayerNeuronCount;
            PrevLayerNeuronCount = prevLayerNeuronCount;

            Gradients = new double[NeuronCount];
            Outputs = new double[NeuronCount];
            Biases = new double[NeuronCount];
            Weights = new double[NeuronCount][];

            // don't do the input layer (neuronCount == 0)
            if (prevLayerNeuronCount > 0)
            {
                for (var n = 0; n < NeuronCount; n++)
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

        public Layer Clone()
        {
            return new Layer
            {
                Weights = Weights.Clone() as double[][],
                Biases = Biases.Clone() as double[],
                Gradients = Gradients.Clone() as double[],
                Outputs = Outputs.Clone() as double[],
                NeuronCount = NeuronCount,
                PrevLayerNeuronCount = PrevLayerNeuronCount,
            };
        }

        private static double GetRandom()
        {
            var x1 = 1 - random.NextDouble();
            var x2 = 1 - random.NextDouble();

            return Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Cos(2.0 * Math.PI * x2);
        }
    }
}
