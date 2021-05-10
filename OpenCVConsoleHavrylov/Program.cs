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
                Console.WriteLine("Press f for face recognition, press t for object tracking, q for exit");
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
                    else if (key.Key == ConsoleKey.T)
                        OpenCVWrapper.TrackObject();
                    else if (key.Key == ConsoleKey.Q)
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
