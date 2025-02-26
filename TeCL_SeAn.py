# app.py

import streamlit as st
from transformers import pipeline

# Load the emotion classification pipeline from Hugging Face (model trained on emotions)
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Define colors and emojis for different emotions
emotion_colors = {
    "joy": "yellow",
    "anger": "red",
    "fear": "purple",
    "sadness": "blue",
    "surprise": "orange",
    "disgust": "green",
    "neutral": "grey"
}

emotion_emoji = {
    "joy": "😊",
    "anger": "😡",
    "fear": "😱",
    "sadness": "😢",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"
}

# Streamlit interface
st.title("Emotion and Sentiment Analysis Tool")
st.markdown("This tool analyzes the sentiment and emotions of the provided text.")

# Input field for text
text_input = st.text_area("Enter Text", "I love this product! It's amazing.")

# Run analysis on the input text
if st.button('Analyze Emotion and Sentiment'):
    # Get emotion classification result
    emotion_result = emotion_pipeline(text_input)
    
    # Extract emotion and score
    emotion_label = emotion_result[0]['label'].lower()
    confidence_score = emotion_result[0]['score']

    # Display the result
    st.subheader("Analysis Result")

    # Display emotion with color and emoji
    emotion_color = emotion_colors.get(emotion_label, "black")
    emotion_emoji_icon = emotion_emoji.get(emotion_label, "🤔")

    # Display emotion with color and emoji
    st.markdown(
        f"<h2 style='color:{emotion_color};'>{emotion_emoji_icon} {emotion_label.capitalize()}</h2>", 
        unsafe_allow_html=True
    )
    st.write(f"Confidence Score: {confidence_score:.2f}")

    # Add explanation based on detected emotion
    if emotion_label == "joy":
        st.write("You're feeling joyful or happy! 😊")
    elif emotion_label == "anger":
        st.write("It seems like you're feeling angry or frustrated! 😡")
    elif emotion_label == "fear":
        st.write("You're feeling fearful or anxious! 😱")
    elif emotion_label == "sadness":
        st.write("You're feeling sad or upset! 😢")
    elif emotion_label == "surprise":
        st.write("You're feeling surprised or amazed! 😲")
    elif emotion_label == "disgust":
        st.write("You're feeling disgusted or repelled! 🤢")
    else:
        st.write("You're feeling neutral or indifferent. 😐")
