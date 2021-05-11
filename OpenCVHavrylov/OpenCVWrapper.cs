using System;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Tracking;
using OpenCvSharp.Dnn;
using System.Windows.Input;
using Tesseract;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Collections.Generic;
using MoreLinq;

namespace OpenCVHavrylov
{
    public class OpenCVWrapper
    {
        public static void Display(Mat im, Point2f[] bbox, string fileName = null)
        {
            for (int i = 0; i < bbox.Count(); i++)
            {
                var j = (i == bbox.Count() - 1 ? 0 : i + 1);
                im.Line(new OpenCvSharp.Point(bbox[i].X, bbox[i].Y), new OpenCvSharp.Point(bbox[j].X, bbox[j].Y), new Scalar(255, 0, 0), 3);
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
                minSize: new OpenCvSharp.Size(30, 30)
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
                    minSize: new OpenCvSharp.Size(30, 30)
                    );


                foreach (var nestedObject in nestedObjects)
                {
                    var center = new OpenCvSharp.Point
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

        public static string ReadText(string inputImagePath)
        {
            var srcImage = new Mat(inputImagePath);
            List<Bitmap> bitmap;
            var bbox = Cv2.SelectROIs("Select Text Boxes. Enter for confirm, Esc for finish", srcImage);
            Cv2.DestroyWindow("Select Text Boxes. Enter for confirm, Esc for finish");
            if (bbox == null || bbox.Length == 0)
                bitmap = new List<Bitmap>() { OpenCvSharp.Extensions.BitmapConverter.ToBitmap(srcImage) };
            else
                bitmap = bbox.Select(x => OpenCvSharp.Extensions.BitmapConverter.ToBitmap(srcImage.SubMat(x))).ToList();

            using (var ocr = new Tesseract.TesseractEngine(@"../../../../OpenCVHavrylov/Data/tessdata", "eng"))
            {
                var res = "";
                int i = 1;
                foreach (var img in bitmap)
                {
                    using (var page = ocr.Process(img))
                    {
                        res += $"[Block {i}]: {page.GetText()}";
                        i++;
                    }
                }
                return res;
            }
        }

        private static Mat Skinmask(Mat image)
        {
            var dest = image.CvtColor(ColorConversionCodes.BGR2HSV);
            var min = new Scalar(0, 48, 80);
            var max = new Scalar(20, 255, 255);
            var skinRegion = dest.InRange(min, max);
            var blurred = skinRegion.Blur(new OpenCvSharp.Size(2, 2));
            return blurred.Threshold(0, 255, ThresholdTypes.Binary);
        }

        private static OpenCvSharp.Point[] GetCNTHull(Mat image, out OpenCvSharp.Point[] contour)
        {
            HierarchyIndex[] hierarchies;
            OpenCvSharp.Point[][] contours;
            image.FindContours(out contours, out hierarchies, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
            contour = contours.MaxBy(x => Cv2.ContourArea(x)).FirstOrDefault();
            OpenCvSharp.Point[] hull = Cv2.ConvexHull(contour);
            return hull;
        }

        private static int[] PointsToIndex(OpenCvSharp.Point[] set, OpenCvSharp.Point[] subset)
        {
            int curSub = 0;
            int[] res = new int[subset.Length];
            for (int i = set.Length - 1; i >= 0; i--)
            {
                if (set[i] == subset[curSub])
                {
                    res[curSub] = i;
                    curSub++;
                }
            }
            return res;
        }

        private static Vec4i[] GetDefects(OpenCvSharp.Point[] contour)
        {
            var hull = Cv2.ConvexHull(contour);
            return Cv2.ConvexityDefects(contour, PointsToIndex(contour, hull));
        }

        public static void DetectHand()
        {
            var cap = new VideoCapture();
            cap.Open(0);
            while (cap.IsOpened())
            {
                try
                {
                    var image = new Mat();
                    cap.Read(image);
                    var mask_img = Skinmask(image);
                    OpenCvSharp.Point[] contour;
                    var hull = GetCNTHull(mask_img, out contour);
                    image.DrawContours(new List<OpenCvSharp.Point[]>() { contour }, -1, new Scalar(255, 255, 0), 2);
                    image.DrawContours(new List<OpenCvSharp.Point[]>() { hull }, -1, new Scalar(0, 255, 255), 2);
                    var defects = GetDefects(contour);
                    if (defects.Length != 0)
                    {
                        var cnt = 0;
                        for (int i = 0; i < defects.Length; i++)
                        {
                            var start = contour[defects[i][0]];
                            var end = contour[defects[i][1]];
                            var far = contour[defects[i][2]];
                            var a = Math.Sqrt(Math.Pow(end.X - start.X, 2) + Math.Pow(end.Y - start.Y, 2));
                            var b = Math.Sqrt(Math.Pow(far.X - start.X, 2) + Math.Pow(far.Y - start.Y, 2));
                            var c = Math.Sqrt(Math.Pow(end.X - far.X, 2) + Math.Pow(end.Y - far.Y, 2));
                            var angle = Math.Acos((b * b + c * c - a * a) / (2 * b * c));
                            if (angle <= Math.PI / 2.0)
                            {
                                cnt++;
                                Cv2.Circle(image, far, 4, new Scalar(0, 0, 255), -1);
                            }

                        }
                        if (cnt > 0)
                            cnt++;
                        image.PutText(cnt.ToString(), new OpenCvSharp.Point(0, 50), HersheyFonts.HersheySimplex, 1, new Scalar(255, 0, 0), 2, LineTypes.AntiAlias);
                    }
                    Cv2.ImShow("Detection", image);
                }
                catch (Exception) { }
                if (Cv2.WaitKey(1) == 113)
                    break;
            }
            cap.Release();
            Cv2.DestroyAllWindows();
        }

        private static void DrawBox(Mat image, Rect2d bbox)
        {
            Cv2.Rectangle(image, rect: new OpenCvSharp.Rect((int)bbox.X, (int)bbox.Y, (int)bbox.Width, (int)bbox.Height),
                new Scalar(255, 0, 255), 3, LineTypes.Link8);
            Cv2.PutText(image, "Tracking", new OpenCvSharp.Point(100, 75), HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 2);
        }

        public static void TrackObject()
        {
            var tracker = TrackerMOSSE.Create();
            var cap = new VideoCapture();
            cap.Open(0);
            var image = new Mat();
            cap.Read(image);
            Cv2.ImShow("Debug", image);
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
                    Cv2.PutText(img, "Lost", new OpenCvSharp.Point(100, 75), HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 2);

                Cv2.Rectangle(img, new OpenCvSharp.Point(15, 15), new OpenCvSharp.Point(200, 90), new Scalar(255, 0, 255), 2);
                Cv2.PutText(img, "Fps:", new OpenCvSharp.Point(20, 40), HersheyFonts.HersheySimplex, 0.7, new Scalar(255, 0, 255), 2);
                Cv2.PutText(img, "Status:", new OpenCvSharp.Point(20, 70), HersheyFonts.HersheySimplex, 0.7, new Scalar(255, 0, 255), 2);


                var fps = (int)(Cv2.GetTickFrequency() / (Cv2.GetTickCount() - timer));
                Scalar myColor;
                if (fps > 60)
                    myColor = new Scalar(20, 230, 20);
                else if (fps > 20)
                    myColor = new Scalar(230, 20, 20);
                else
                    myColor = new Scalar(20, 20, 230);
                Cv2.PutText(img, fps.ToString(), new OpenCvSharp.Point(75, 40), HersheyFonts.HersheySimplex, 0.7, myColor, 2);

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
