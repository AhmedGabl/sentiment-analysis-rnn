import streamlit as st
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Set the page configuration
st.set_page_config(
    page_title="Sentiment Analysis App", 
    page_icon=":smiley:", 
    layout="wide"
)

# Cache the model and tokenizer loading for efficiency
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("artifacts/model.keras")
    with open("artifacts/tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Load resources
model, tokenizer = load_model_and_tokenizer()

# Define max length for padding
MAX_LENGTH = 100

# Function to preprocess user input
def preprocess_text(text, tokenizer, max_length):
    tokens = tokenizer.texts_to_sequences([text])
    padded_tokens = pad_sequences(tokens, maxlen=max_length, padding='post')
    return padded_tokens

# Custom CSS for styling
st.markdown("""
    <style>
        .stApp {
            background-color: #1e1e2f; /* Darker background for the entire app */
            color: #f5f5f5; /* Light text color */
        }
        .stTitle h1 {
            color: #ff4b4b; /* Title in red */
            text-align: center;
        }
        .stMarkdown h2 {
            color: #4b4bff; /* Subheading in blue */
        }
        .stTextArea textarea {
            border-radius: 10px;
            font-size: 16px;
            background-color: #3e3e3e; /* Darker text input background */
            color: #f5f5f5;
        }
        .stButton>button {
            background-color: #f5c518; /* IMDb yellow */
            color: black;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 10px;
        }
        .result-box {
            padding: 20px;
            background-color: #2e2e2e; /* Box for results */
            border-radius: 10px;
            text-align: center;
        }
        .positive {
            color: #4caf50; /* Green for positive sentiment */
        }
        .negative {
            color: #f44336; /* Red for negative sentiment */
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.title(":smiley: Sentiment Analysis App")
st.write("""
    This app predicts whether a movie review has a **positive** or **negative** sentiment.  
    Enter a review below to see the prediction in action!
""")

# Input box for user text
st.subheader("Enter Your Movie Review")
user_input = st.text_area(
    "Type your review below:", 
    placeholder="e.g., 'The movie was amazing! I loved every moment of it.'"
)

# Details about the data and processing pipeline
with st.expander("🔍 How Does It Work?"):
    st.write("""
        - **Input:** The app accepts a movie review in plain text.  
        - **Processing:**  
          1. Converts the text into tokens using a pre-trained tokenizer.  
          2. Pads the tokens to a fixed length of 100 words for consistency.  
        - **Prediction:** The pre-trained RNN model predicts the sentiment.  
          - Output is a probability value.  
          - If probability > 0.5, the sentiment is **Positive**; otherwise, **Negative**.  
    """)

# Prediction button and logic
if st.button("Predict Sentiment"):
    if user_input.strip():
        # Preprocess input
        with st.spinner("Processing the input and making predictions..."):
            processed_input = preprocess_text(user_input, tokenizer, MAX_LENGTH)
            prediction = model.predict(processed_input)
            sentiment = "Positive" if prediction > 0.5 else "Negative"
            confidence = prediction[0][0]
        
        # Display prediction results
        st.subheader("Prediction Results")
        st.markdown(f"""
            <div class='result-box'>
                <h3>Sentiment: <span class='{sentiment.lower()}'>{sentiment}</span></h3>
                <p><strong>Confidence:</strong> {confidence:.2f}</p>
            </div>
        """, unsafe_allow_html=True)
    else:   
        st.warning("⚠️ Please enter a valid review!")

# Footer
st.markdown("---")
st.write("💡 Built with [Streamlit](https://streamlit.io/) | Pre-trained model using IMDb movie reviews.")
