import streamlit as st
import base64
import os
from diffusers import DiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from torchaudio import load, save
import torchaudio
import io

# Create audio_output folder if it doesn't exist
output_folder = 'audio_output'
os.makedirs(output_folder, exist_ok=True)

pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1")
pipe = pipe.to("mps")
params = SpectrogramParams()
converter = SpectrogramImageConverter(params)

def predict_and_generate_audio(prompt, output_folder='audio_output'):
    spec = pipe(
        prompt,
        width=768,
    ).images[0]

    wav = converter.audio_from_spectrogram_image(image=spec)
    output_path = os.path.join(output_folder, 'output.wav')
    wav.export(output_path, format='wav')
    return output_path

def main():
    st.title('Text to Music Generator')
    
    st.write("Generate music from text prompts using diffusion models. Enter a prompt to get started!")
    
    with st.form(key='audio_form'):
        prompt = st.text_input('Enter your text prompt:')
        submit_button = st.form_submit_button(label='Generate Music')
        
        if submit_button:
            if not prompt:
                st.error('Prompt cannot be empty.')
            else:
                with st.spinner('Generating audio...'):
                    audio_path = predict_and_generate_audio(prompt)
                    
                    # Load audio data into a tensor
                    waveform, sample_rate = torchaudio.load(audio_path)
                    
                    # Specify the output audio file name
                    file_name = "music.wav"
                    output_audio_path = os.path.join("audio_output", file_name) 
                    
                    # Save the audio data
                    torchaudio.save(output_audio_path, waveform, sample_rate)
                    
                    # Convert audio file to base64
                    with open(output_audio_path, "rb") as audio_file:
                        base64_bytes = base64.b64encode(audio_file.read()).decode('utf-8')
                    
                    st.success("Audio successfully generated and saved!")
                    st.audio(output_audio_path)

if __name__ == '__main__':
    main()
