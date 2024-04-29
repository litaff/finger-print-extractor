namespace FingerPrintExtractorModule
{
    using System;
    using UnityEngine;
    using UnityEngine.UI;

    [Serializable]
    public class TextureButtonEntry
    {
        public Texture2D texture;
        public Button button;
        
        public event Action<Texture2D> OnClick;

        public void Initialize()
        {
            button.onClick.AddListener(OnClickHandler);
        }

        private void OnClickHandler()
        {
            OnClick?.Invoke(texture);
        }
    }
}