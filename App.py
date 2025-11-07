import streamlit as st
import librosa, numpy as np, soundfile as sf, noisereduce as nr, math, io
from pydub import AudioSegment

st.set_page_config(page_title="Audiobook Fixer 🎧", page_icon="🎶", layout="centered")

st.title("🎧 Audiobook Fixer")
st.write("""
Upload your MP3 chapters, and this app will:
- Remove background noise (choose Mild or Strong)
- Normalize RMS to -21 dB (ACX standard)
- Add 3 seconds of silence at the start and 4 seconds at the end
- Return ACX-compliant MP3s ready for upload!
""")

nr_level = st.radio("Noise reduction level:", ["Mild", "Strong"])
uploaded_files = st.file_uploader("Upload one or more MP3 files", type=["mp3"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} file(s)... Please wait ⏳")
    progress_bar = st.progress(0)
    total = len(uploaded_files)

    for idx, file in enumerate(uploaded_files, start=1):
        # Load audio
        y, sr = librosa.load(file, sr=None, mono=True)

        # Noise reduction
        prop = 0.5 if nr_level == "Mild" else 1.0
        y_nr = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop, stationary=False)

        # Trim silence and pad 3s start / 4s end
        intervals = librosa.effects.split(y_nr, top_db=40)
        core = y_nr[intervals[0, 0]:intervals[-1, 1]] if intervals.size > 0 else y_nr
        y_fixed = np.concatenate([np.zeros(int(sr * 3)), core, np.zeros(int(sr * 4))])

        # Normalize to -21 dB RMS
        current_rms = 20 * math.log10(np.sqrt(np.mean(y_fixed ** 2)))
        gain = 10 ** ((-21 - current_rms) / 20)
        y_final = y_fixed * gain

        # Export directly to MP3 in memory
        buf = io.BytesIO()
        sf.write(buf, y_final.astype(np.float32), sr, format='WAV')
        buf.seek(0)
        mp3_buf = io.BytesIO()
        AudioSegment.from_file(buf, format="wav").export(mp3_buf, format="mp3", bitrate="192k")
        mp3_buf.seek(0)

        # Create download button
        st.download_button(
            label=f"⬇️ Download {file.name.replace('.mp3', '_processed.mp3')}",
            data=mp3_buf,
            file_name=file.name.replace(".mp3", "_processed.mp3"),
            mime="audio/mpeg"
        )

        progress_bar.progress(idx / total)

    st.success("✅ All files processed!")
