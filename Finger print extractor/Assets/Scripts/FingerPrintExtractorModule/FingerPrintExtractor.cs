namespace FingerPrintExtractorModule
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using OpenCVForUnity.CoreModule;
    using OpenCVForUnity.ImgprocModule;
    using OpenCVForUnity.UnityUtils;
    using Sirenix.OdinInspector;
    using UnityEngine;

    public class FingerPrintExtractor : MonoBehaviour
    {
        private const string CONVERT = "Convert";
        private const string NORMALIZE = "Normalize";
        private const string ADAPTIVE_THRESHOLD = "Adaptive Threshold";
        private const string GAUSSIAN_BLUR = "Gaussian Blur";
        private const string GABOR_FILTER = "Gabor Filter";
        private const string THRESHOLD = "Threshold";
        private const string CLOSING = "Closing";
        private const string DISPLAY = "Display";
        private const string DATABASE = "Database";

        public ComputeShader Cs;
        public WebCameraController WebCameraController;

        [SerializeField, BoxGroup(DATABASE)]
        private List<TextureButtonEntry> textureButtons;

        [SerializeField, BoxGroup(DISPLAY)]
        private Renderer previewDisplay;
        [SerializeField, BoxGroup(DISPLAY)]
        private Renderer maskDisplay;
        [SerializeField, BoxGroup(DISPLAY)]
        private Renderer outputDisplay;

        [SerializeField, BoxGroup(CONVERT)]
        private float contrast1 = .8f;
        [SerializeField, BoxGroup(CONVERT)]
        private float brightness1 = 25f;
        [SerializeField, BoxGroup(NORMALIZE)]
        private float normalizeAlpha = 0f;
        [SerializeField, BoxGroup(NORMALIZE)]
        private float normalizeBeta = 255f;
        [SerializeField, BoxGroup(ADAPTIVE_THRESHOLD)]
        private int blockSize1 = 67;
        [SerializeField, BoxGroup(ADAPTIVE_THRESHOLD)]
        private int c1 = 2;
        [SerializeField, BoxGroup(GAUSSIAN_BLUR)]
        private float sigmaX1 = 5f;
        [SerializeField, BoxGroup(CONVERT)]
        private float contrast2 = 1.7f;
        [SerializeField, BoxGroup(CONVERT)]
        private float brightness2 = -40f;
        [SerializeField, BoxGroup(GAUSSIAN_BLUR)]
        private float sigmaX2 = 5f;
        [SerializeField, BoxGroup(ADAPTIVE_THRESHOLD)]
        private int blockSize2 = 67;
        [SerializeField, BoxGroup(ADAPTIVE_THRESHOLD)]
        private int c2 = 2;
        [SerializeField, BoxGroup(THRESHOLD)]
        private float thresh = 128f; // just lower than maxValue
        [SerializeField, BoxGroup(THRESHOLD)]
        private float maxValue = 255f;
        [SerializeField, BoxGroup(CLOSING)]
        private int closingKernelSize = 1; // maybe expose this to the user as form of a slider

        [SerializeField, BoxGroup(GABOR_FILTER)]
        private int filterAmount = 3; // lower faster
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private Size filterSize = new(20, 20);
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private double sigma = 3;
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private double lambda = 100;
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private double gamma = 0; // in color gamma sense
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private double psi = 1;
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private int depth = -1;

        private Texture2D outputTexture;
        private Texture2D maskTexture;

        private void Awake()
        {
            WebCameraController.StartWebCam(new Vector2(10000, 10000));
            foreach (var textureButton in textureButtons)
            {
                textureButton.Initialize();
                textureButton.OnClick += UpdateTexture;
            }
        }

        private void Update()
        {
            if (!WebCameraController.gameObject.activeSelf) return;

            var inputTexture = WebCameraController.WebCamTexture;

            if (!inputTexture.didUpdateThisFrame) return;

            if (inputTexture == null) return;

            var inputMat = new Mat(inputTexture.height, inputTexture.width, CvType.CV_8UC4);
            var outputMat = new Mat(inputTexture.height, inputTexture.width, CvType.CV_8UC4);

            Utils.webCamTextureToMat(inputTexture, inputMat);

            outputTexture = new Texture2D(inputTexture.width, inputTexture.height, TextureFormat.RGBA32, false);
            maskTexture = new Texture2D(inputTexture.width, inputTexture.height, TextureFormat.RGBA32, false);

            previewDisplay.material.mainTexture = inputTexture;
            maskDisplay.material.mainTexture = maskTexture;
            outputDisplay.material.mainTexture = outputTexture;

            var features = ExtractFeatures(inputMat);
            var mask = ExtractMaskFromFeatures(features);
            Core.bitwise_and(features, mask, outputMat);

            Utils.matToTexture2D(mask, maskTexture);
            Utils.matToTexture2D(features, outputTexture);
        }

        private void UpdateTexture(Texture2D inputTexture)
        {
            outputTexture = new Texture2D(inputTexture.width, inputTexture.height, TextureFormat.RGBA32, false);
            maskTexture = new Texture2D(inputTexture.width, inputTexture.height, TextureFormat.RGBA32, false);
            previewDisplay.material.mainTexture = inputTexture;
            maskDisplay.material.mainTexture = maskTexture;
            outputDisplay.material.mainTexture = outputTexture;

            var inputMat = new Mat(inputTexture.height, inputTexture.width, CvType.CV_8UC4);
            var outputMat = new Mat(inputTexture.height, inputTexture.width, CvType.CV_8UC4);
            Utils.texture2DToMat(inputTexture, inputMat);

            var features = ExtractFeatures(inputMat);
            var mask = ExtractMaskFromFeatures(features);
            Core.bitwise_and(features, mask, outputMat);

            Utils.matToTexture2D(mask, maskTexture);
            Utils.matToTexture2D(outputMat, outputTexture);
        }

        private List<Mat> CreateGaborFilter()
        {
            List<Mat> filters = new();

            for (float theta = 0; theta < Mathf.PI; theta += Mathf.PI / filterAmount)
            {
                var kernel = Imgproc.getGaborKernel(filterSize, sigma, theta, lambda, gamma, psi, CvType.CV_64F);
                kernel /= 1 * kernel.total();
                filters.Add(kernel);
            }

            return filters;
        }

        private Mat ApplyGaborFilters(Mat input, List<Mat> filters)
        {
            var newImage = new Mat(input.rows(), input.cols(), CvType.CV_8UC1);
            var imageFilter = new Mat(input.rows(), input.cols(), CvType.CV_8UC1);

            foreach (var filter in filters)
            {
                Imgproc.filter2D(input, imageFilter, depth, filter);
                Core.max(newImage, imageFilter, newImage);
            }

            return newImage;
        }

        private MatOfPoint GetLargestContour(List<MatOfPoint> contours)
        {
            var largestContour = contours.OrderByDescending(Imgproc.contourArea).First();
            return largestContour;
        }

        private Mat ExtractFeatures(Mat input)
        {
            var result = input.clone();
            // Adjust contrast and brightness
            Core.convertScaleAbs(result, result, contrast1, brightness1);
            // Convert to grayscale
            Imgproc.cvtColor(result, result, Imgproc.COLOR_RGBA2GRAY);
            // Apply histogram equalization - contrast equalization based on the image's histogram
            Imgproc.equalizeHist(result, result);
            // Normalize the grayscale image to a range of alpha to beta (deemed not necessary)
            //Core.normalize(result, result, normalizeAlpha, normalizeBeta, Core.NORM_MINMAX);

            // Applied adaptive binary thresholding of type Gaussian, meaning weighted sum
            // maxValue is 255 cause we want to convert to binary, full black-white
            // blockSize - size of the pixel neighborhood that is used to calculate a threshold value for the pixel, must be an odd number
            // c - constant subtracted from the mean or weighted mean
            Imgproc.adaptiveThreshold(result, result, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY,
                blockSize1, c1);

            /*var tempTexture = new Texture2D(result.width(), result.height(), TextureFormat.RGBA32, false);
            Utils.matToTexture2D(result, tempTexture);
            AdaptiveThreshold(tempTexture, blockSize1, c1);
            Utils.texture2DToMat(tempTexture, result);*/

            Imgproc.GaussianBlur(result, result, new Size(5, 5), sigmaX1);
            // Adjust contrast and brightness
            Core.convertScaleAbs(result, result, contrast2, brightness2);
            Imgproc.GaussianBlur(result, result, new Size(5, 5), sigmaX2);
            // Create and apply Gabor filters - edge detection
            var gaborFilters = CreateGaborFilter();
            result = ApplyGaborFilters(result, gaborFilters);
            // Apply histogram equalization - contrast equalization based on the image's histogram
            Imgproc.equalizeHist(result, result);
            // Applied adaptive binary thresholding of type Gaussian, meaning weighted sum
            // maxValue is 255 cause we want to convert to binary, full black-white
            // blockSize - size of the pixel neighborhood that is used to calculate a threshold value for the pixel, must be an odd number
            // c - constant subtracted from the mean or weighted mean
            Imgproc.adaptiveThreshold(result, result, 255,
                Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY,
                blockSize2, c2);

            /*tempTexture = new Texture2D(result.width(), result.height(), TextureFormat.RGBA32, false);
            Utils.matToTexture2D(result, tempTexture);
            AdaptiveThreshold(tempTexture, blockSize2, c2);
            Utils.texture2DToMat(tempTexture, result);*/
            return result;
        }

        private Mat ExtractMaskFromFeatures(Mat features)
        {
            var mask = new Mat(features.height(), features.width(), CvType.CV_8UC1);
            // Find contours using https://www.nevis.columbia.edu/~vgenty/public/suzuki_et_al.pdf
            // Imgproc.CHAIN_APPROX_SIMPLE - compresses horizontal, vertical, and diagonal segments and leaves
            // only their end points, might be the most memory efficient
            // Imgproc.RETR_EXTERNAL - retrieves only the extreme outer contours, might be the most memory efficient
            var contours = new List<MatOfPoint>();
            var hierarchy = new Mat(features.height(), features.width(), CvType.CV_8UC1);
            Imgproc.findContours(features, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            // Protect from empty contours
            if (contours.Count == 0) return features;

            // Select contour with the largest area
            var largestContour = GetLargestContour(contours);

            // Draw contour on mask, -1 means draw all contours and -1 means fill the contour
            Imgproc.drawContours(mask, new List<MatOfPoint> {largestContour}, -1,
                new Scalar(255), -1);
            return mask;
        }

        private void AdaptiveThreshold(Texture renderTexture, int ksize, int c)
        {
            // adaptive threshold with gaussian kernel

            if (ksize % 2 == 0) ksize++;
            Cs.SetInt("ksize", ksize);

            var computeGaussianKernelKernel = Cs.FindKernel("ComputeGaussianKernel");
            var gaussianWeightsBuffer = new ComputeBuffer(ksize * ksize, sizeof(float));
            gaussianWeightsBuffer.SetData(new float[ksize * ksize]);
            var sigma = (float) (0.3 * ((ksize - 1) * 0.5 - 1) + 0.8);
            Cs.SetFloat("sigma", sigma);

            var border = (ksize - 1) / 2;
            Cs.SetInt("border", border);
            Cs.SetBuffer(computeGaussianKernelKernel, "kernel", gaussianWeightsBuffer);
            Cs.Dispatch(computeGaussianKernelKernel, ksize, ksize, 1);

            var gaussianAdaptiveThresholdKernel = Cs.FindKernel("GaussianAdaptiveThreshold");
            var inputTexture = new RenderTexture(renderTexture.width, renderTexture.height, 1)
            {
                enableRandomWrite = true
            };
            inputTexture.Create();
            Cs.SetBuffer(gaussianAdaptiveThresholdKernel, "kernel", gaussianWeightsBuffer);
            Cs.SetTexture(gaussianAdaptiveThresholdKernel, "input", renderTexture);
            Cs.SetTexture(gaussianAdaptiveThresholdKernel, "result", inputTexture);
            Cs.SetInt("c", c);
            Cs.Dispatch(gaussianAdaptiveThresholdKernel, renderTexture.width / 32,
                renderTexture.height / 32,
                1);

            gaussianWeightsBuffer.Release();
        }
    }
}