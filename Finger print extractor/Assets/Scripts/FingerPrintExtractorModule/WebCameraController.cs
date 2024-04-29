namespace FingerPrintExtractorModule
{
    using System;
    using System.Threading.Tasks;
    using UnityEngine;

    public class WebCameraController : MonoBehaviour
    {
        public WebCamTexture WebCamTexture { get; private set; }

        /// <summary>
        /// Invoked after web cam is found and is playing
        /// </summary>
        public event Action OnWebCamFound;
        /// <summary>
        /// Invoked when a web cam is not found
        /// </summary>
        public event Action OnWebCamNotFound;
        /// <summary>
        /// Invoked when user denies permission to access a web cam
        /// </summary>
        public event Action OnAccessDenied;

        /// <summary>
        /// Call this if you want to start working with the camera
        /// </summary>
        /// <param name="scanZone"></param>
        /// <returns></returns>
        public async void StartWebCam(Vector2 scanZone)
        {
            await SetUpDevices(scanZone);
            if (!WebCamTexture)
            {
                OnWebCamNotFound?.Invoke();
                return;
            }
            WebCamTexture.Play();
            OnWebCamFound?.Invoke();
        }

        /// <summary>
        /// Call this after you're done with the camera
        /// </summary>
        public void StopWebCam()
        {
            WebCamTexture.Stop();
        }

        /// <summary>
        /// Will setup webcam and try to use scan zone, will use the closest aspect ratio that the camera supports
        /// example: PC - 512x512 => 640x360, Mobile - 256x256 => 256x256
        /// If the user denies permission, the user has to manually give permissions to the web cam  
        /// </summary>
        private Task SetUpDevices(Vector2 scanZone)
        {
            if (WebCamTexture) return Task.CompletedTask;
            
            Application.RequestUserAuthorization(UserAuthorization.WebCam);
            
            if (Application.HasUserAuthorization(UserAuthorization.WebCam))
            {
                var devices = WebCamTexture.devices;

                foreach (var webCam in devices)
                {
                    // check if this is the only camera, if so the it without question
                    if (devices.Length > 1)
                        if (webCam.isFrontFacing) continue;

                    // take the first valid camera and break the loop
                    WebCamTexture = new WebCamTexture(
                        webCam.name, (int)scanZone.y, (int)scanZone.x);
                    break;
                }
            }
            else
            {
                OnAccessDenied?.Invoke();
            }

            return Task.CompletedTask;
        }
    }
}