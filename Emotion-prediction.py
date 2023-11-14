import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import neattext.functions as nfx
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
from _cffi_backend import callback

# Load your trained model
pipe_lr = joblib.load(open("model/text_emotion_new.pkl", "rb"))

# Define emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "joy": "😂", "neutral": "😐", "sadness": "😔", "surprise": "😮"
}

# Initialize the session state
session_state = st.session_state

# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Function to capture voice input
def capture_voice():
    st.write("Speak something:")
    duration = 10  # seconds
    filename = "temp_audio.wav"

    with sd.OutputStream(callback=callback, channels=2):
        audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=2, dtype='int16')
        sd.wait()

    # Save audio data to a temporary file
    sf.write(filename, audio_data, 44100)

    # Recognize speech from the saved audio file
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(filename)

    with audio_file as source:
        try:
            text = recognizer.record(source)
            result = recognizer.recognize_google(text)
            st.write("Text from Speech:", result)

            # Predict emotions for the recognized text
            prediction = predict_emotions(result)
            probability = get_prediction_proba(result)

            # Display the predicted emotion
            st.subheader("Predicted Emotion")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction, emoji_icon))
            st.write("Confidence:{}".format(np.max(probability)))

            # Set session state to indicate voice input was captured
            session_state.voice_captured = True

        except sr.UnknownValueError:
            st.write("Sorry, could not understand audio.")
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")

    # Remove the temporary audio file
    st.file_uploader("Upload the audio file", type=["wav"])

# Streamlit app
def main():
    # Initialize session state variables
    if 'voice_captured' not in session_state:
        session_state.voice_captured = False

    st.title("Emotional Detection From Text")
    st.subheader("Detect Emotions In Text")

    # If voice input is not captured, display the voice input option
    if not session_state.voice_captured:
        voice_input = st.checkbox("Enable Voice Input")

        if voice_input:
            capture_voice()
            # Use st.experimental_rerun() to trigger a rerun without a button
            st.experimental_rerun()

        else:
            with st.form(key='my_form'):
                raw_text = st.text_area("Type Here")
                submit_text = st.form_submit_button(label='Submit')

            # Display results if text is submitted
            if submit_text:
                col1, col2 = st.columns(2)

                # Preprocess the input text if it is not None
                if 'raw_text' in locals() and raw_text is not None:
                    clean_text = nfx.remove_userhandles(raw_text)
                    clean_text = nfx.remove_stopwords(clean_text)

                    # Predict emotions for the entire paragraph
                    prediction = predict_emotions(clean_text)
                    probability = get_prediction_proba(clean_text)

                    with col1:
                        st.success("Original Text")
                        st.write(raw_text)

                        st.success("Prediction")
                        emoji_icon = emotions_emoji_dict[prediction]
                        st.write("{}:{}".format(prediction, emoji_icon))
                        st.write("Confidence:{}".format(np.max(probability)))

                    with col2:
                        st.success("Prediction Probability")
                        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                        proba_df_clean = proba_df.T.reset_index()
                        proba_df_clean.columns = ["emotions", "probability"]

                        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                        st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
