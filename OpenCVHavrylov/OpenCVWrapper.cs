using System;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Tracking;
using OpenCvSharp.Dnn;
using System.Windows.Input;

namespace OpenCVHavrylov
{
    public class OpenCVWrapper
    {
        public static void Display(Mat im, Point2f[] bbox, string fileName = null)
        {
            for (int i = 0; i < bbox.Count(); i++)
            {
                var j = (i == bbox.Count() - 1 ? 0 : i + 1);
                im.Line(new Point(bbox[i].X, bbox[i].Y), new Point(bbox[j].X, bbox[j].Y), new Scalar(255, 0, 0), 3);
            }
            var resultName = fileName == null ? $"{Directory.GetCurrentDirectory()}/result.png" : $"{fileName}result.{fileName.Split('.').Last()}";
            im.ImWrite(resultName);
        }
        public static string DecodeQR(string inputImagePath)
        {
            Mat inputImage = new Mat(inputImagePath);
            QRCodeDetector qrDecoder = new QRCodeDetector();
            Point2f[] point2Fs;
            string data = qrDecoder.DetectAndDecode(inputImage, out point2Fs);
            if (data.Length > 0)

            {

                Display(inputImage, point2Fs, inputImagePath);
                inputImage.Dispose();
                qrDecoder.Dispose();
                return data;

            }

            else
            {
                throw new Exception("QR Not Found");
                inputImage.Dispose();
                qrDecoder.Dispose();
            }

        }
        public static int RecognizeFaces(string inputImagePath)
        {
            var srcImage = new Mat(inputImagePath);

            var grayImage = new Mat();
            Cv2.CvtColor(srcImage, grayImage, ColorConversionCodes.BGRA2GRAY);
            Cv2.EqualizeHist(grayImage, grayImage);
            var cascade = new CascadeClassifier(@"..\..\..\..\OpenCVHavrylov\Data\haarcascade_frontalface_alt.xml");
            var nestedCascade = new CascadeClassifier(@"..\..\..\..\OpenCVHavrylov\Data\haarcascade_eye_tree_eyeglasses.xml");
            var faces = cascade.DetectMultiScale(
                image: grayImage,
                scaleFactor: 1.1,
                minNeighbors: 2,
                flags: HaarDetectionType.DoRoughSearch | HaarDetectionType.ScaleImage,
                minSize: new Size(30, 30)
                );

            var count = 1;
            foreach (var faceRect in faces)
            {
                var detectedFaceImage = new Mat(srcImage, faceRect);

                var color = Scalar.FromRgb(255, 0, 0);
                Cv2.Rectangle(srcImage, faceRect, color, 3);


                var detectedFaceGrayImage = new Mat();
                Cv2.CvtColor(detectedFaceImage, detectedFaceGrayImage, ColorConversionCodes.BGRA2GRAY);
                var nestedObjects = nestedCascade.DetectMultiScale(
                    image: detectedFaceGrayImage,
                    scaleFactor: 1.1,
                    minNeighbors: 2,
                    flags: HaarDetectionType.DoRoughSearch | HaarDetectionType.ScaleImage,
                    minSize: new Size(30, 30)
                    );


                foreach (var nestedObject in nestedObjects)
                {
                    var center = new Point
                    {
                        X = (int)(Math.Round(nestedObject.X + nestedObject.Width * 0.5, MidpointRounding.ToEven) + faceRect.Left),
                        Y = (int)(Math.Round(nestedObject.Y + nestedObject.Height * 0.5, MidpointRounding.ToEven) + faceRect.Top)
                    };
                    var radius = Math.Round((nestedObject.Width + nestedObject.Height) * 0.25, MidpointRounding.ToEven);
                    Cv2.Circle(srcImage, center, (int)radius, color, thickness: 3);
                }

                count++;
            }
            var resultName = inputImagePath == null ? $"{Directory.GetCurrentDirectory()}/result.png" : $"{inputImagePath}result.{inputImagePath.Split('.').Last()}";
            srcImage.ImWrite(resultName);
            srcImage.Dispose();
            return faces.Length;
        }

        private static void DrawBox(Mat image, Rect2d bbox)
        {
            Cv2.Rectangle(image, rect: new Rect((int)bbox.X, (int)bbox.Y, (int)bbox.Width, (int)bbox.Height),
                new Scalar(255, 0, 255), 3, LineTypes.Link8);
            Cv2.PutText(image, "Tracking", new Point(100, 75), HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 2);
        }

        public static void TrackObject()
        {
            var tracker = TrackerMOSSE.Create();
            var cap = new VideoCapture();
            cap.Open(0);
            var image = new Mat();
            cap.Read(image);
            var bbox = Cv2.SelectROI("Tracking", image, false);
            var bbox2d = new Rect2d(bbox.X, bbox.Y, bbox.Width, bbox.Height);
            tracker.Init(image, bbox2d);
            while (true)
            {

                var timer = Cv2.GetTickCount();
                var img = new Mat();
                var success = cap.Read(img);
                success = tracker.Update(img, ref bbox2d);

                if (success)
                    DrawBox(img, bbox2d);
                else
                    Cv2.PutText(img, "Lost", new Point(100, 75), HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 2);

                Cv2.Rectangle(img, new Point(15, 15), new Point(200, 90), new Scalar(255, 0, 255), 2);
                Cv2.PutText(img, "Fps:", new Point(20, 40), HersheyFonts.HersheySimplex, 0.7, new Scalar(255, 0, 255), 2);
                Cv2.PutText(img, "Status:", new Point(20, 70), HersheyFonts.HersheySimplex, 0.7, new Scalar(255, 0, 255), 2);


                var fps = (int)(Cv2.GetTickFrequency() / (Cv2.GetTickCount() - timer));
                Scalar myColor;
                if (fps > 60)
                    myColor = new Scalar(20, 230, 20);
                else if (fps > 20)
                    myColor = new Scalar(230, 20, 20);
                else
                    myColor = new Scalar(20, 20, 230);
                Cv2.PutText(img, fps.ToString(), new Point(75, 40), HersheyFonts.HersheySimplex, 0.7, myColor, 2);

                Cv2.ImShow("Tracking", img);
                if (Cv2.WaitKey(1) == 113)
                {
                    Cv2.DestroyWindow("Tracking");
                    break;
                }
            }
        }
    }
}
