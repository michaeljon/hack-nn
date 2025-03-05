using System;
using System.IO;
using System.Threading.Tasks;
using MessagePack;

namespace Learning.Neural.Networks
{
    internal static class NetworkSerializer
    {
        public static void SaveNetwork(NeuralNetwork network, string folder, int epoch, double accuracy, string layerName)
        {
            var path = Path.Join(folder, $"L-{layerName}-E-{epoch}-A-{accuracy:P}-{DateTime.Now:s}.bin");

            if (Directory.Exists(folder) == false)
            {
                Directory.CreateDirectory(folder);
            }

            var files = Directory.GetFiles(folder, "*.bin");

            foreach (var file in files)
            {
                var filename = Path.GetFileName(file);

                if (filename.Contains(layerName))
                {
                    var fileAccuracy = double.Parse(filename.Split("-")[5].Split("%")[0]) / 100.0;

                    if (accuracy > fileAccuracy)
                    {
                        Task.Run(async () => { await FileDelete(Path.Join(folder, filename)); });
                    }
                    else
                    {
                        Console.WriteLine("There is an existing file with a better accuracy for this network configuration");
                        return;
                    }
                }
            }

            using (var fileStream = new FileStream(path, FileMode.Create))
            {
                try
                {
                    MessagePackSerializer.Serialize(fileStream, network);
                    Console.WriteLine("Network configuration saved");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to write network configuration {ex.Message}");
                    Task.Run(async () => { await FileDelete(path); });
                }
            }
        }

        private static async Task FileDelete(string path)
        {
            var retryDelay = 100;
            int retryCount = 0;
            var deleted = false;

            while (deleted == false && retryCount < 100)
            {
                try
                {
                    File.Delete(path);
                    deleted = true;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error deleting {path}");
                    Console.WriteLine(ex.Message);
                    await Task.Delay(retryDelay);
                }
            }

            if (deleted == true)
            {
                Console.WriteLine($"File {path} deleted");
            }
            else
            {
                Console.WriteLine($"File {path} not deleted");
            }
        }
    }
}
