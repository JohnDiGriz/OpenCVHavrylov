using System;
using OpenCVHavrylov;
using System.IO;
using System.Linq;

namespace OpenCVConsoleHavrylov
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                Console.WriteLine("Enter Image Path: ");
                var path = Console.ReadLine();
                Console.WriteLine("Press q to detect QrCode, or f to detect face");
                var key = Console.ReadKey();
                Console.WriteLine();
                try
                {
                    if (key.Key == ConsoleKey.Q)
                    {
                        Console.WriteLine($"QR Data: {OpenCVWrapper.DecodeQR(path)}");
                    }
                    else if(key.Key == ConsoleKey.F)
                    {
                        Console.WriteLine($"Found faces: {OpenCVWrapper.RecognizeFaces(path)}");
                    }
                    var result = path == null ? $"{Directory.GetCurrentDirectory()}/result.png" : $"{path}result.{path.Split('.').Last()}";
                    Console.WriteLine($"Resulting image saved as {result}");
                    Console.WriteLine("Press any key");
                    Console.ReadKey();
                }
                catch(Exception ex)
                {
                    Console.WriteLine($"Error: {ex.Message}");
                    Console.WriteLine("Press any key");
                    Console.ReadKey();
                }
            }

        }
    }
}
