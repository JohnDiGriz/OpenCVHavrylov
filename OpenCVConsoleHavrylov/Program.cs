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
                Console.WriteLine("Press f for face recognition\nPress t for object tracking\nPress r for text recognition\nPress h for hand recognition\nPress q for QR Code\nPress e for exit");
                var key = Console.ReadKey();
                Console.WriteLine();

                try
                {
                    if (key.Key == ConsoleKey.F)
                    {
                        Console.WriteLine("Enter Image Path: ");
                        var path = Console.ReadLine();
                        Console.WriteLine($"Found faces: {OpenCVWrapper.RecognizeFaces(path)}");
                        var result = path == null ? $"{Directory.GetCurrentDirectory()}/result.png" : $"{path}result.{path.Split('.').Last()}";
                        Console.WriteLine($"Resulting image saved as {result}");
                    }
                    else if (key.Key == ConsoleKey.R)
                    {
                        Console.WriteLine("Enter Image Path: ");
                        var path = Console.ReadLine();
                        Console.WriteLine($"Text:\n {OpenCVWrapper.ReadText(path)}");
                    }
                    else if (key.Key == ConsoleKey.T)
                        OpenCVWrapper.TrackObject();
                    else if (key.Key == ConsoleKey.H)
                        OpenCVWrapper.DetectHand();
                    else if (key.Key == ConsoleKey.Q)
                    {
                        Console.WriteLine("Enter Image Path: ");
                        var path = Console.ReadLine();
                        Console.WriteLine($"Result:\n {OpenCVWrapper.DecodeQR(path)}");
                    }
                    else if (key.Key == ConsoleKey.E)
                        break;
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
