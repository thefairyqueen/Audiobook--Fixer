import streamlit as st
import io
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional noise reduction
try:
    import noisereduce as nr
except ImportError:
    nr = None
    st.warning("noisereduce not installed, noise reduction will be disabled.")

# Audio handling with safe fallback
try:
    from pydub import AudioSegment
    from pydub.utils import make_chunks
except ImportError:
    AudioSegment = None
    make_chunks = None
    st.error("pydub not installed, audio features will not work.")

# Page config
st.set_page_config(page_title="Audiobook Fixer", layout="wide")

st.title("Audiobook Fixer 🎧")

# Upload audio
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "ogg"])

if uploaded_file and AudioSegment:
    try:
        audio = AudioSegment.from_file(io.BytesIO(uploaded_file.read()))
        st.audio(uploaded_file, format='audio/wav')
        
        st.success(f"Loaded audio: {len(audio)}ms long.")

        # Example: Split into chunks
        chunk_length_ms = 30000  # 30 sec
        chunks = make_chunks(audio, chunk_length_ms)

        st.write(f"Split audio into {len(chunks)} chunks.")

        # Optional: visualize waveform (requires numpy & matplotlib)
        samples = np.array(audio.get_array_of_samples())
        plt.figure(figsize=(10, 3))
        plt.plot(samples)
        plt.title("Waveform")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        st.pyplot(plt)
        
    except Exception as e:
        st.error(f"Failed to process audio: {e}")

else:
    if not AudioSegment:
        st.warning("Audio functionality disabled due to missing pydub.")
    st.info("Upload an audio file to get started.")
    rms = 20 * np.log10(np.sqrt(np.mean(audio ** 2)))
    gain = 10 ** ((target_db - rms) / 20)
    return audio * gain

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} file(s)... Please wait ⏳")
    progress_bar = st.progress(0)
    total = len(uploaded_files)

    processed_files = []

    for idx, file in enumerate(uploaded_files, start=1):
        with st.spinner(f"Processing {file.name}..."):
            # Load full MP3 with pydub
            audio = AudioSegment.from_file(file, format="mp3")

            # Split into 30s chunks for memory efficiency
            chunk_length_ms = 30000
            chunks = make_chunks(audio, chunk_length_ms)

            # Start with 3s silence
            processed_audio = AudioSegment.silent(duration=3000, frame_rate=44100)

            prop = 0.5 if nr_level == "Mild" else 1.0

            for chunk in chunks:
                # Convert to numpy array
                samples = np.array(chunk.get_array_of_samples()).astype(np.float32)

                # Reshape if stereo
                if chunk.channels == 2:
                    samples = samples.reshape((-1, 2)).T  # Shape: (2, n_samples)
                else:
                    samples = samples[np.newaxis, :]  # Shape: (1, n_samples)

                # Apply noise reduction and normalization per channel
                processed_channels = []
                for channel in samples:
                    channel_nr = nr.reduce_noise(y=channel, sr=chunk.frame_rate, prop_decrease=prop, stationary=False)
                    channel_nr = normalize_rms(channel_nr, target_db=-21.0)
                    processed_channels.append(channel_nr)

                # Recombine channels
                processed_chunk = np.vstack(processed_channels).T if len(processed_channels) > 1 else processed_channels[0]

                # Convert back to AudioSegment (ACX standard: 16-bit, 44.1 kHz)
                processed_segment = AudioSegment(
                    processed_chunk.astype(np.int16).tobytes(),
                    frame_rate=44100,
                    sample_width=2,  # 16-bit PCM
                    channels=len(processed_channels)
                )
                processed_audio += processed_segment

            # Add 4s silence at the end
            processed_audio += AudioSegment.silent(duration=4000, frame_rate=44100)

            # Export full processed MP3 to in-memory
            tmp_mp3 = io.BytesIO()
            processed_audio.export(tmp_mp3, format="mp3", bitrate="192k")
            tmp_mp3.seek(0)

            mp3_name = file.name.replace(".mp3", "_processed.mp3")
            processed_files.append((mp3_name, tmp_mp3))

            # Export first 10 seconds for preview
            preview_segment = processed_audio[:10000]  # first 10 seconds
            tmp_preview = io.BytesIO()
            preview_segment.export(tmp_preview, format="mp3", bitrate="192k")
            tmp_preview.seek(0)

            # Display preview audio player
            st.audio(tmp_preview, format="audio/mp3", start_time=0, caption=f"Preview: {file.name}")

            progress_bar.progress(idx / total)

    st.success("✅ All files processed!")

    # Download buttons
    for name, audio_data in processed_files:
        st.download_button(
            label=f"⬇️ Download {name}",
            data=audio_data,
            file_name=name,
            mime="audio/mpeg"
        )



