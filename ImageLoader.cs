using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Learning.Neural.Networks
{
    internal static class ImageLoader
    {
        public static (byte[][] X, int[] y) LoadImages(string path)
        {
            var files = Directory.GetFiles(path, "*.png").AsEnumerable();
            byte[][] imageData = new byte[files.Count()][];

            files.Each((file, index) =>
            {
                LoadImageFromFile(file, index, imageData);
            });

            return (imageData, null);
        }

        public static (Matrix<double> X, Matrix<double> y) LoadImagesAsMatrix(string path, int count = 100)
        {
            var files = Directory
                .GetFiles(path, "*.png", SearchOption.AllDirectories)
                .AsEnumerable()
                .OrderBy(s => Guid.NewGuid().ToString("n"))
                .Take(count);

            byte[][] imageData = new byte[files.Count()][];
            double[] labels = new double[files.Count()];

            files.Each((file, index) =>
            {
                labels[index] = double.Parse(Path.GetDirectoryName(file).Split(Path.DirectorySeparatorChar).Last());

                LoadImageFromFile(file, index, imageData);
            });

            var X = Matrix<double>.Build.Dense(files.Count(), 784, (i, j) => Convert.ToDouble(imageData[i][j]));
            var y = Matrix<double>.Build.Dense(files.Count(), 1, (i, j) => labels[i]);

            return (X, y);
        }

        static void LoadImageFromFile(string file, int index, byte[][] imageData)
        {
            var image = Image.Load<L8>(file);

            imageData[index] = new byte[784];

            image.CopyPixelDataTo(imageData[index]);
        }
    }

    internal static class Iterators
    {
        public static void Each<T>(this IEnumerable<T> enumerable, Action<T, int> action)
        {
            var index = 0;

            foreach (var item in enumerable)
            {
                action(item, index++);
            }
        }
    }
}
