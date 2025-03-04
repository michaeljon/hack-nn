using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Learning.Neural.Networks
{
    internal static class ImageLoader
    {
        public static byte[][] LoadImages(string path)
        {
            var files = Directory.GetFiles(path, "*.png").AsEnumerable();
            byte[][] imageData = new byte[files.Count()][];

            files.Each((file, index) =>
            {
                LoadImageFromFile(file, index, imageData);
            });

            return imageData;
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
