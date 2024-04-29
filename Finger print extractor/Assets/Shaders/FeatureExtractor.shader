Shader "Custom/FeatureExtractor"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _HistTex ("Histogram Texture", 2D) = "white" {}
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            half3 adjust_contrast_curve(half3 color, half contrast)
            {
                return pow(abs(color * 2 - 1), 1 / max(contrast, 0.0001)) * sign(color - 0.5) + 0.5;
            }

            half3 convert_scale_abs(half3 color, half alpha, half beta)
            {
                return abs(color * alpha + beta/255);
            }
            
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            #define NUM_BINS 256
            sampler2D _MainTex;
            sampler2D _HistTex;
            static float hist[NUM_BINS];
            static float cdf[NUM_BINS];
            
            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 col = tex2D(_MainTex, i.uv);

                col.rgb = convert_scale_abs(col.rgb, 0.8, 0);

                // Convert to grayscale
                float gray = dot(col.rgb, float3(0.299, 0.587, 0.114));

                hist[int(gray * (NUM_BINS - 1))] = gray;

                cdf[0] = hist[0];
                for (int j = 1; j < NUM_BINS; j++)
                {
                    cdf[j] = cdf[j - 1] + hist[j];
                }

                gray = cdf[int(gray * (NUM_BINS - 1))];
                
                return fixed4(gray, gray, gray, col.a);
            }
            ENDCG
        }
    }
}
