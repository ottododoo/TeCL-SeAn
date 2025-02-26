import streamlit as st
from transformers import pipeline

# Load the emotion classification pipeline using a free model
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

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
    emotion_result = emotion_pipeline(text_input, truncation=True)
    
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
    emotion_messages = {
        "joy": "You're feeling joyful or happy! 😊",
        "anger": "It seems like you're feeling angry or frustrated! 😡",
        "fear": "You're feeling fearful or anxious! 😱",
        "sadness": "You're feeling sad or upset! 😢",
        "surprise": "You're feeling surprised or amazed! 😲",
        "disgust": "You're feeling disgusted or repelled! 🤢",
        "neutral": "You're feeling neutral or indifferent. 😐"
    }

    st.write(emotion_messages.get(emotion_label, "Emotion not recognized. 🤔"))
