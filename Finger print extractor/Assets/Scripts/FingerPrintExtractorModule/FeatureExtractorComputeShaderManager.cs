namespace FingerPrintExtractorModule
{
    using System.Collections.Generic;
    using System.Linq;
    using Sirenix.OdinInspector;
    using UnityEngine;

    public class FeatureExtractorComputeShaderManager : MonoBehaviour
    {
        private const string COMPUTE_SETTINGS = "Compute Settings";
        private const string GABOR_FILTER = "Gabor Filter";
        private const string INPUT = "Input";

        [SerializeField, BoxGroup(INPUT)]
        private WebCameraController webCameraController;
        [SerializeField, BoxGroup(INPUT)]
        private ComputeShader extractFeatures;
        [SerializeField, BoxGroup(INPUT)]
        private Renderer display;
        [SerializeField, BoxGroup(INPUT)]
        private Texture2D finger;

        [SerializeField, BoxGroup(COMPUTE_SETTINGS)]
        private float contrast1 = 0.8f;
        [SerializeField, BoxGroup(COMPUTE_SETTINGS)]
        private float brightness1 = 25f;
        [SerializeField, BoxGroup(COMPUTE_SETTINGS)]
        private int thresholdKsize = 3;
        [SerializeField, BoxGroup(COMPUTE_SETTINGS)]
        private int c = 0;
        [SerializeField, BoxGroup(COMPUTE_SETTINGS)]
        private int blurKsize = 5;
        [SerializeField, BoxGroup(COMPUTE_SETTINGS)]
        private float blurSigma = 5;
        [SerializeField, BoxGroup(COMPUTE_SETTINGS)]
        private float contrast2 = 1.7f;
        [SerializeField, BoxGroup(COMPUTE_SETTINGS)]
        private float brightness2 = -40f;

        [SerializeField, BoxGroup(GABOR_FILTER)]
        private int filterAmount = 3; // lower faster
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private int filterSize = 20;
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private float sigma = 3;
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private float lambda = 100;
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private float gamma = 0; // in color gamma sense
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private float psi = 1;
        [SerializeField, BoxGroup(GABOR_FILTER)]
        private int depth = -1;

        private RenderTexture renderTexture;

        public List<RenderTexture> GaborFilter;

        private void Awake()
        {
            webCameraController.StartWebCam(new Vector2(10000, 10000));
        }

        private void Update()
        {
            if (!webCameraController.gameObject.activeSelf) return;

            var inputTexture = webCameraController.WebCamTexture;

            if (!inputTexture.didUpdateThisFrame) return;

            if (inputTexture == null) return;

            InitializeRenderTextures(inputTexture);
            ExecuteShader();

            display.sharedMaterial.mainTexture = renderTexture;
        }

        [Button("Compute")]
        private void Compute()
        {
            InitializeRenderTextures(finger);
            ExecuteShader();
            display.sharedMaterial.mainTexture = renderTexture;
        }

        private void InitializeRenderTextures(Texture inputTexture)
        {
            if (renderTexture != null) renderTexture.Release();
            renderTexture = new RenderTexture(inputTexture.width, inputTexture.height, 1)
            {
                enableRandomWrite = true
            };
            renderTexture.Create();
            Graphics.Blit(inputTexture, renderTexture);
        }

        private void ExecuteShader()
        {
            // Adjust contrast and brightness
            ConvertScaleAbs(contrast1, brightness1);

            // Compute histogram from grayscale image
            var histogram = GrayScaleHist();

            // Equalize histogram
            var equalizedHistogram = EqualizeHistogram(histogram);

            // Apply equalized histogram
            ApplyEqualizedHistogram(equalizedHistogram);

            // Adaptive threshold with gaussian kernel
            AdaptiveThresholdingWithGaussianWeights(GetGaussianKernel(thresholdKsize));

            // Gaussian blur
            GaussianBlur(GetGaussianKernel(blurKsize, blurSigma));

            // Adjust contrast and brightness
            ConvertScaleAbs(contrast2, brightness2);

            // Gaussian blur
            GaussianBlur(GetGaussianKernel(blurKsize, blurSigma));

            // Gabor filter
            GaborFilter = GetGaborFilter(filterAmount, filterSize, sigma, lambda, gamma, psi).ToList();
        }

        private ComputeBuffer GetGaussianKernel(int ksize, float sigma = 0f)
        {
            if (ksize < 3) ksize = 3;
            if (ksize % 2 == 0) ksize++;
            if (sigma == 0f) sigma = 0.3f * ((ksize - 1) * 0.5f - 1) + 0.8f;
            var computeGaussianKernelKernel = extractFeatures.FindKernel("ComputeGaussianKernel");

            var gaussianWeightsBuffer = new ComputeBuffer(ksize * ksize, sizeof(float));
            gaussianWeightsBuffer.SetData(new float[ksize * ksize]);
            extractFeatures.SetFloat("sigma", sigma);
            extractFeatures.SetInt("ksize", ksize);
            var border = (ksize - 1) / 2;
            extractFeatures.SetInt("border", border);
            extractFeatures.SetBuffer(computeGaussianKernelKernel, "kernel", gaussianWeightsBuffer);
            extractFeatures.Dispatch(computeGaussianKernelKernel, ksize, ksize, 1);

            return gaussianWeightsBuffer;
        }

        private void AdaptiveThresholdingWithGaussianWeights(ComputeBuffer gaussianKernel)
        {
            var gaussianAdaptiveThresholdKernel = extractFeatures.FindKernel("GaussianAdaptiveThreshold");

            var inputTexture = new RenderTexture(renderTexture.width, renderTexture.height, 1)
            {
                enableRandomWrite = true
            };
            inputTexture.Create();
            Graphics.Blit(renderTexture, inputTexture);
            extractFeatures.SetBuffer(gaussianAdaptiveThresholdKernel, "kernel", gaussianKernel);
            extractFeatures.SetTexture(gaussianAdaptiveThresholdKernel, "input", inputTexture);
            extractFeatures.SetTexture(gaussianAdaptiveThresholdKernel, "result", renderTexture);
            extractFeatures.SetInt("c", c);
            extractFeatures.Dispatch(gaussianAdaptiveThresholdKernel, renderTexture.width / 32,
                renderTexture.height / 32,
                1);

            gaussianKernel.Release();
            RenderTexture.active = renderTexture;
            inputTexture.Release();
        }

        private void GaussianBlur(ComputeBuffer gaussianKernel)
        {
            var gaussianBlurKernel = extractFeatures.FindKernel("GaussianBlur");
            var inputTexture = new RenderTexture(renderTexture.width, renderTexture.height, 1)
            {
                enableRandomWrite = true
            };
            inputTexture.Create();
            Graphics.Blit(renderTexture, inputTexture);
            extractFeatures.SetBuffer(gaussianBlurKernel, "kernel", gaussianKernel);
            extractFeatures.SetTexture(gaussianBlurKernel, "input", inputTexture);
            extractFeatures.SetTexture(gaussianBlurKernel, "result", renderTexture);
            extractFeatures.Dispatch(gaussianBlurKernel, renderTexture.width / 32,
                renderTexture.height / 32,
                1);

            gaussianKernel.Release();
            RenderTexture.active = renderTexture;
            inputTexture.Release();
        }

        private void ConvertScaleAbs(float alpha, float beta)
        {
            var convertScaleAbsKernel = extractFeatures.FindKernel("ConvertScaleAbs");

            extractFeatures.SetFloat("alpha", alpha);
            extractFeatures.SetFloat("beta", beta);
            extractFeatures.SetTexture(convertScaleAbsKernel, "result", renderTexture);
            extractFeatures.Dispatch(convertScaleAbsKernel, renderTexture.width / 8, renderTexture.height / 8, 1);
        }

        private ComputeBuffer GrayScaleHist()
        {
            var scaleGrayHistKernel = extractFeatures.FindKernel("GrayScaleHist");

            var histogramBuffer = new ComputeBuffer(256, sizeof(int));
            histogramBuffer.SetData(new int[256]);
            extractFeatures.SetBuffer(scaleGrayHistKernel, "histogram", histogramBuffer);
            extractFeatures.SetTexture(scaleGrayHistKernel, "result", renderTexture);

            extractFeatures.Dispatch(scaleGrayHistKernel, renderTexture.width / 8, renderTexture.height / 8, 1);

            return histogramBuffer;
        }

        private ComputeBuffer EqualizeHistogram(ComputeBuffer histogramBuffer)
        {
            var equalizeHistogramKernel = extractFeatures.FindKernel("EqualizeHist");
            var equalizeHistogram = new ComputeBuffer(256, sizeof(float));
            equalizeHistogram.SetData(new float[256]);

            extractFeatures.SetBuffer(equalizeHistogramKernel, "histogram", histogramBuffer);
            extractFeatures.SetBuffer(equalizeHistogramKernel, "cdf", equalizeHistogram);
            extractFeatures.SetInt("width", renderTexture.width);
            extractFeatures.SetInt("height", renderTexture.height);
            extractFeatures.Dispatch(equalizeHistogramKernel, 1, 1, 1);
            histogramBuffer.Release();

            return equalizeHistogram;
        }

        private void ApplyEqualizedHistogram(ComputeBuffer equalizedHistogram)
        {
            var applyEqualizedHistogramKernel = extractFeatures.FindKernel("ApplyEqualizedHist");

            extractFeatures.SetBuffer(applyEqualizedHistogramKernel, "cdf", equalizedHistogram);
            extractFeatures.SetTexture(applyEqualizedHistogramKernel, "result", renderTexture);
            extractFeatures.Dispatch(applyEqualizedHistogramKernel, renderTexture.width / 8, renderTexture.height / 8,
                1);

            equalizedHistogram.Release();
        }

        private RenderTexture[] GetGaborFilter(int filterAmount, int ksize, float sigma, float lambda, float gamma,
            float psi)
        {
            var gaborFilter = new RenderTexture[filterAmount];
            var computeGaborKernelKernel = extractFeatures.FindKernel("ComputeGaborKernel");
            extractFeatures.SetFloat("sigma", sigma);
            extractFeatures.SetFloat("lambda", lambda);
            extractFeatures.SetFloat("gamma", gamma);
            extractFeatures.SetFloat("psi", psi);

            var index = 0;
            for (float theta = 0; theta < Mathf.PI; theta += Mathf.PI / filterAmount)
            {
                var filterTexture = new RenderTexture(ksize, ksize, 1)
                {
                    enableRandomWrite = true
                };
                filterTexture.Create();

                extractFeatures.SetFloat("theta", theta);
                extractFeatures.SetTexture(computeGaborKernelKernel, "gaborKernel", filterTexture);
                extractFeatures.Dispatch(computeGaborKernelKernel, ksize, ksize, 1);
                gaborFilter[index] = filterTexture;
                index++;

                /*for (int x = 0; x < ksize; x++)
                {
                    for (int y = 0; y < ksize; y++)
                    {
                        float total = ksize * ksize;
                        Vector2 rotatedUV = new Vector2();
                        rotatedUV.x = x * Mathf.Cos(theta) - y * Mathf.Sin(theta);
                        rotatedUV.y = x * Mathf.Sin(theta) + y * Mathf.Cos(theta);

                        float value = Mathf.Exp(-(rotatedUV.x * rotatedUV.x + gamma + rotatedUV.y * rotatedUV.y) /
                                                (2 * sigma * sigma))
                                      * Mathf.Cos((float) (2 * 3.14 * rotatedUV.x / lambda + psi));

                        //value /= 1 * total;
                        Debug.Log(value);
                    }
                }*/
            }

            return gaborFilter;
        }
    }
}