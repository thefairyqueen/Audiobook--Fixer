import streamlit as st
import numpy as np
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.utils import make_chunks
import io

# Page config
st.set_page_config(
    page_title="Audiobook Fixer 🎧",
    page_icon="🎶",
    layout="centered"
)

st.title("🎧 Audiobook Fixer")
st.write("""
Upload your MP3 chapters, and this app will:
- Remove background noise (choose Mild or Strong)
- Normalize RMS to -21 dB (ACX standard)
- Add 3 seconds of silence at the start and 4 seconds at the end
- Return ACX-compliant MP3s ready for upload!
- Preview the first 10 seconds before downloading
""")

# User inputs
nr_level = st.radio("Noise reduction level:", ["Mild", "Strong"])
uploaded_files = st.file_uploader(
    "Upload one or more MP3 files",
    type=["mp3"],
    accept_multiple_files=True
)

def normalize_rms(audio: np.ndarray, target_db=-21.0) -> np.ndarray:
    """Normalize RMS per channel to target dB"""
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
                    channel_nr = nr.reduce_noise(
                        y=channel,
                        sr=chunk.frame_rate,
                        prop_decrease=prop,
                        stationary=False
                    )
                    channel_nr = normalize_rms(channel_nr, target_db=-21.0)
                    processed_channels.append(channel_nr)

                # Recombine channels
                if len(processed_channels) > 1:
                    processed_chunk = np.vstack(processed_channels).T
                else:
                    processed_chunk = processed_channels[0]

                # Convert back to AudioSegment (ACX standard: 16-bit, 44.1 kHz)
                processed_segment = AudioSegment(
                    data=processed_chunk.astype(np.int16).tobytes(),
                    sample_width=2,       # 16-bit PCM
                    frame_rate=44100,
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
            mime="audio/mp3"
        )
            data=audio_data,
            file_name=name,
            mime="audio/mp3")
    return audio * gain

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} file(s)... Please wait ⏳")
    progress_bar = st.progress(0)
    total = len(uploaded_files)
    processed_files = []

    for idx, file in enumerate(uploaded_files, start=1):
        with st.spinner(f"Processing {file.name}..."):
            audio = AudioSegment.from_file(file, format="mp3")
            chunk_length_ms = 30000
            chunks = make_chunks(audio, chunk_length_ms)

            processed_audio = AudioSegment.silent(duration=3000, frame_rate=44100)
            prop = 0.5 if nr_level == "Mild" else 1.0

            for chunk in chunks:
                samples = np.array(chunk.get_array_of_samples()).astype(np.float32)
                if chunk.channels == 2:
                    samples = samples.reshape((-1, 2)).T
                else:
                    samples = samples[np.newaxis, :]

                processed_channels = []
                for channel in samples:
                    channel_nr = nr.reduce_noise(
                        y=channel, 
                        sr=chunk.frame_rate, 
                        prop_decrease=prop, 
                        stat
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
                    channel_nr = nr.reduce_noise(
                        y=channel, sr=chunk.frame_rate, prop_decrease=prop, stationary=False)
                    channel_nr = normalize_rms(channel_nr, target_db=-21.0)
                    processed_channels.append(channel_nr)

                # Recombine channels
                processed_chunk = (np.vstack(processed_channels).T
                    if len(processed_channels) > 1
                    else processed_channels[0])

                # Convert back to AudioSegment (ACX standard: 16-bit, 44.1 kHz)
                processed_segment = AudioSegment(
                    processed_chunk.astype(np.int16).tobytes(),
                    frame_rate=44100,
                    sample_width=2,  # 16-bit PCM
                    channels=len(processed_channels)
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
            mime="audio/mpeg")

def normalize_rms(audio: np.ndarray, target_db=-21.0) -> np.ndarray:
    """Normalize RMS per channel to target dB"""
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
                    channel_nr = nr.reduce_noise(
                        y=channel,
                        sr=chunk.frame_rate,
                        prop_decrease=prop,
                        stationary=False)
                    channel_nr = normalize_rms(channel_nr, target_db=-21.0)
                    processed_channels.append(channel_nr)

                # Recombine channels
                processed_chunk = (
                    np.vstack(processed_channels).T
                    if len(processed_channels) > 1
                    else processed_channels[0])

                # Convert back to AudioSegment (ACX standard: 16-bit, 44.1 kHz)
                processed_segment = AudioSegment(
                    processed_chunk.astype(np.int16).tobytes(),
                    frame_rate=44100,
                    sample_width=2,
                    channels=len(processed_channels)
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
            preview_segment = processed_audio[:10000]
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
            mime="audio/mpeg")








