import gradio as gr
import os
import glob
import numpy as np
from helper import load_text_to_speech, load_voice_style

# Configuration
ONNX_DIR = "assets/onnx"
VOICE_STYLES_DIR = "assets/voice_styles"
DEFAULT_SPEED = 1.05
DEFAULT_STEPS = 5

# Supported languages in Supertonic 2
SUPPORTED_LANGUAGES = [
    ("English", "en"),
    ("Korean", "ko"),
    ("Spanish", "es"),
    ("Portuguese", "pt"),
    ("French", "fr"),
]

# Global model instance
tts_model = None

def get_voice_styles():
    """Scan the voice styles directory for JSON files."""
    if not os.path.exists(VOICE_STYLES_DIR):
        return []
    files = glob.glob(os.path.join(VOICE_STYLES_DIR, "*.json"))
    # Return filenames only for the dropdown
    return [os.path.basename(f) for f in files]

def load_model():
    """Initialize the TTS model if not already loaded."""
    global tts_model
    if tts_model is None:
        if not os.path.exists(ONNX_DIR):
            raise FileNotFoundError(f"ONNX directory not found at '{ONNX_DIR}'. Please ensure assets are correctly linked.")
        print(f"Loading TTS model from {ONNX_DIR}...")
        tts_model = load_text_to_speech(ONNX_DIR, use_gpu=False)
        print("Model loaded successfully.")
    return tts_model

def generate_audio(text, voice_style_name, language, speed, total_steps):
    try:
        if not text.strip():
            return None, "Please enter some text."
        
        if not voice_style_name:
            return None, "Please select a voice style."

        model = load_model()
        
        voice_style_path = os.path.join(VOICE_STYLES_DIR, voice_style_name)
        if not os.path.exists(voice_style_path):
            return None, f"Voice style file not found: {voice_style_path}"

        # Load the selected voice style
        style = load_voice_style([voice_style_path])

        # Generate speech with language parameter (Supertonic 2 API)
        wav, duration = model(text, language, style, total_steps, speed)
        
        # Prepare audio for Gradio
        audio_data = wav.flatten()
        sample_rate = model.sample_rate
        
        return (sample_rate, audio_data), f"Generated successfully. Duration: {duration.sum():.2f}s"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error: {str(e)}"

# Create the Gradio Interface
with gr.Blocks(title="Supertonic 2 TTS WebUI") as demo:
    gr.Markdown("# Supertonic 2 TTS WebUI")
    gr.Markdown("Generate multilingual speech from text using ONNX Runtime. Supports English, Korean, Spanish, Portuguese, and French.")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Input Text",
                lines=5,
                placeholder="Enter text here...",
                value="This morning, I took a walk in the park, and the sound of the birds and the breeze was so pleasant that I stopped for a long time just to listen."
            )
            
            # Dynamic voice style loading
            available_styles = get_voice_styles()
            default_style = available_styles[0] if available_styles else None
            
            voice_style_dropdown = gr.Dropdown(
                choices=available_styles,
                value=default_style,
                label="Voice Style",
                info="Select a voice style JSON file from assets/voice_styles"
            )
            
            # Language selection for Supertonic 2
            language_dropdown = gr.Dropdown(
                choices=[lang[1] for lang in SUPPORTED_LANGUAGES],
                value="en",
                label="Language",
                info="Select the language of your input text"
            )
            
            with gr.Row():
                speed_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=DEFAULT_SPEED,
                    step=0.05,
                    label="Speed",
                    info="Speech speed (higher = faster)"
                )
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=DEFAULT_STEPS,
                    step=1,
                    label="Denoising Steps",
                    info="More steps might improve quality but take longer"
                )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Generated Audio")
            status_output = gr.Textbox(label="Status", interactive=False)

    # Event handlers
    generate_btn.click(
        fn=generate_audio,
        inputs=[text_input, voice_style_dropdown, language_dropdown, speed_slider, steps_slider],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch()
