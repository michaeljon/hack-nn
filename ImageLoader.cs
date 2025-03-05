using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Learning.Neural.Networks
{
    public static class ImageLoader
    {
        private const string TrainImages = "train-images.idx3-ubyte";
        private const string TrainLabels = "train-labels.idx1-ubyte";
        private const string TestImages = "t10k-images.idx3-ubyte";
        private const string TestLabels = "t10k-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(TestImages, TestLabels))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            using var labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            using var images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            var magicNumber = images.ReadBigInt32();
            var numberOfImages = images.ReadBigInt32();
            var width = images.ReadBigInt32();
            var height = images.ReadBigInt32();

            var magicLabel = labels.ReadBigInt32();
            var numberOfLabels = labels.ReadBigInt32();

            for (var i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var label = labels.ReadByte();

                yield return new Image(bytes, label);
            }
        }
    }

    public class Image
    {
        public int Label { get; set; }

        public double[] Targets { get; set; } = new double[10];

        public double[] Data { get; set; }

        public Image(byte[] data, byte label)
        {
            Data = [.. data.Select(b => b / 256d)];
            Label = Convert.ToInt32(label);
            Targets[Label] = 1;
        }
    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(int));

            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(bytes);
            }

            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
