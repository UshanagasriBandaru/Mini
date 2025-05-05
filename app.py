import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import shap
import psycopg2
import logging
from datetime import datetime, timedelta
import time
import random
import uuid
from textblob import TextBlob
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mindmend.log'),
        logging.StreamHandler()
    ]
)

# Database configuration
DB_HOST = "localhost"
DB_NAME = "mindmenddb"
DB_USER = "postgres"
DB_PASSWORD = "usha"

# Database initialization
def init_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                points INTEGER DEFAULT 0,
                is_admin BOOLEAN DEFAULT FALSE,
                last_login TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS mood_tracker (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(user_id),
                created_at TIMESTAMP,
                mood TEXT,
                notes TEXT
            );
            CREATE TABLE IF NOT EXISTS interactions (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(user_id),
                created_at TIMESTAMP,
                input_text TEXT,
                response TEXT,
                emotion TEXT,
                attention_scores TEXT,
                feedback TEXT,
                interaction_id TEXT
            );
            CREATE TABLE IF NOT EXISTS moments (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(user_id),
                created_at TIMESTAMP,
                moment TEXT,
                category TEXT
            );
            CREATE TABLE IF NOT EXISTS achievements (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(user_id),
                achievement TEXT,
                points INTEGER,
                created_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS game_scores (
                id SERIAL PRIMARY KEY,
                user_id TEXT REFERENCES users(user_id),
                game_name TEXT,
                score INTEGER,
                created_at TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_users ON users(user_id);
            CREATE INDEX IF NOT EXISTS idx_mood ON mood_tracker(user_id);
            CREATE INDEX IF NOT EXISTS idx_interactions ON interactions(user_id);
            CREATE INDEX IF NOT EXISTS idx_game_scores ON game_scores(user_id);
        """)
        conn.commit()
        check_and_migrate_db(conn)
        logging.info("Database initialized successfully")
        return conn
    except psycopg2.Error as e:
        logging.error(f"PostgreSQL error: {e.pgcode} - {e.pgerror}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during database initialization: {e}")
        return None

# Schema migration check
def check_and_migrate_db(conn):
    try:
        c = conn.cursor()
        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'moments' AND column_name = 'category'")
        if not c.fetchone():
            c.execute("ALTER TABLE moments ADD COLUMN category TEXT")
            conn.commit()
            logging.info("Added missing 'category' column to moments table")
        c.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'interactions' AND column_name = 'interaction_id'")
        if not c.fetchone():
            c.execute("ALTER TABLE interactions ADD COLUMN interaction_id TEXT")
            conn.commit()
            logging.info("Added missing 'interaction_id' column to interactions table")
    except psycopg2.Error as e:
        logging.error(f"Error during schema migration: {e.pgcode} - {e.pgerror}")
        conn.rollback()

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None, None

# Fallback emotion detection
def detect_emotion_fallback(text):
    text = text.lower()
    if any(phrase in text for phrase in ["not in good mood", "not good", "feeling down", "bad mood", "sad", "unhappy"]):
        return "sadness"
    elif any(word in text for word in ["happy", "great", "awesome"]):
        return "joy"
    elif any(word in text for word in ["stress", "anxious", "worried"]):
        return "fear"
    elif any(word in text for word in ["angry", "mad", "frustrated"]):
        return "anger"
    return "neutral"

# Text analysis
def analyze_text(text, chat_history):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    keywords = [word for word in text.lower().split() if len(word) > 3]
    return {"sentiment": sentiment, "keywords": keywords}

# Generate motivational response
def generate_motivational_response(text, emotion, chat_history, username, analysis):
    responses = {
        "sadness": [
            f"Hey {username}, I'm sorry you're feeling down. Want to talk about what's been tough?",
            f"It's okay to feel low sometimes, {username}. Would you like a calming exercise or to share more?"
        ],
        "joy": [
            f"That's awesome, {username}! What's got you in such a great mood?",
            f"Love hearing you're happy, {username}! Want to share more or play a quick game?"
        ],
        "fear": [
            f"Hey {username}, stress can be heavy. Try breathing: in 4, hold 4, out 4. What's on your mind?",
            f"I sense some worry, {username}. Let's talk it through or try a grounding exercise."
        ],
        "anger": [
            f"Oof, sounds frustrating, {username}. Want to vent or try a quick relaxation trick?",
            f"Anger’s tough, {username}. Let’s count to 10 together. What happened?"
        ],
        "neutral": [
            f"Hey {username}, thanks for sharing! How can I support you today?",
            f"Hi {username}! What's on your mind? Feeling like a chat or a fun activity?"
        ]
    }
    if any(keyword in text.lower() for keyword in ["mental health", "therapy", "counseling", "depression", "anxiety"]):
        return f"Hey {username}, I'm here to help with mental health questions. Could you share more details about {text}? For example, are you looking for coping tips or just want to talk?"
    return random.choice(responses.get(emotion, responses["neutral"]))

# Explain response
def explain_response(text, emotion, shap_scores, tokenizer, chat_history):
    explanation = []
    explanation.append(f"I detected {emotion.capitalize()} in your message.")
    explanation.append("**Why I think this**:")
    
    if shap_scores:
        explanation.append("I used a DistilBERT model to analyze your text, focusing on the meaning and context of your words.")
        important_words = [word for word, score in shap_scores.items() if score > 0]
        if important_words:
            explanation.append(f"**Key words that stood out**: {', '.join(important_words)}")
            if st.session_state.explanation_mode.lower() == "technical":
                explanation.append("**Technical note**: The model assigns importance scores (SHAP values) to words based on their contribution to the emotion prediction.")
    else:
        explanation.append("I used a keyword-based approach to detect the emotion.")
        keywords = [word for word in text.lower().split() if word in ["sad", "happy", "stress", "angry"]]
        if keywords:
            explanation.append(f"**Detected keywords**: {', '.join(keywords)}")
    
    sentiment = TextBlob(text).sentiment.polarity
    explanation.append(f"**Tone of your message**: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")
    
    if st.session_state.explanation_mode.lower() in ["detailed", "technical"] and chat_history:
        recent_emotions = [msg["emotion"] for msg in chat_history[-5:] if msg.get("emotion")]
        if recent_emotions:
            explanation.append(f"**Recent context**: Your previous messages showed emotions like {', '.join(set(recent_emotions))}.")
    
    if st.session_state.explanation_mode.lower() == "technical":
        explanation.append("**Model details**: DistilBERT is a lightweight transformer model trained on sentiment data, fine-tuned for emotion detection.")
        if shap_scores:
            explanation.append("**SHAP analysis**: Each word’s influence is calculated using SHAP (SHapley Additive exPlanations) to show how it contributes to the predicted emotion.")
    
    explanation = [str(item) for item in explanation if item]
    logging.info(f"Generated explanation: {explanation}")
    return "\n".join(explanation)

# Check achievements
def check_achievements(user_id, conn, action, created_at):
    try:
        c = conn.cursor()
        c.execute("SELECT achievement FROM achievements WHERE user_id = %s", (user_id,))
        existing_achievements = [row[0] for row in c.fetchall()]
        new_achievements = []
        points = 0
        if action == "mood_log" and "First Mood Log" not in existing_achievements:
            new_achievements.append(("First Mood Log", 10))
        elif action == "chat" and "First Chat" not in existing_achievements:
            new_achievements.append(("First Chat", 10))
        elif action == "moment_shared" and "First Moment Shared" not in existing_achievements:
            new_achievements.append(("First Moment Shared", 10))
        elif action == "game_completed" and "First Game Played" not in existing_achievements:
            new_achievements.append(("First Game Played", 10))
        for achievement, pts in new_achievements:
            with conn:
                c.execute(
                    "INSERT INTO achievements (user_id, achievement, points, created_at) VALUES (%s, %s, %s, %s)",
                    (user_id, achievement, pts, created_at)
                )
                c.execute("UPDATE users SET points = points + %s WHERE user_id = %s", (pts, user_id))
                points += pts
                logging.info(f"Awarded {pts} points for {achievement} to user {user_id}")
        conn.commit()
        logging.info(f"Total points awarded: {points} for user {user_id}")
        return points
    except psycopg2.Error as e:
        logging.error(f"Database error checking achievements: {e.pgcode} - {e.pgerror}")
        conn.rollback()
        return 0
    except Exception as e:
        logging.error(f"Unexpected error checking achievements: {e}")
        conn.rollback()
        return 0

# Generate response with model
def generate_response(text, tokenizer, model, user_id, conn, chat_history):
    shap_scores = {}
    explainer = None
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    interaction_id = str(uuid.uuid4())
    username = get_username(user_id, conn)

    if tokenizer is None or model is None:
        logging.warning("Model unavailable, using fallback emotion detection")
        emotion = detect_emotion_fallback(text)
        response = generate_motivational_response(text, emotion, chat_history, username, analyze_text(text, chat_history))
        try:
            c = conn.cursor()
            with conn:
                c.execute(
                    "INSERT INTO interactions (user_id, created_at, input_text, response, emotion, interaction_id) VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, created_at, text, response, emotion, interaction_id)
                )
                conn.commit()
            check_achievements(user_id, conn, "chat", created_at)
        except psycopg2.Error as e:
            logging.error(f"Database error saving interaction: {e.pgcode} - {e.pgerror}")
            conn.rollback()
        return response, emotion, shap_scores, created_at, explainer, interaction_id

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=-1).item()
        emotions = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise", 6: "neutral"}
        emotion = emotions[predicted_label]
        if "not in good mood" in text.lower() or "not good" in text.lower():
            emotion = "sadness"
            logging.info("Emotion override applied for 'not in good mood' to sadness")
        response = generate_motivational_response(text, emotion, chat_history, username, analyze_text(text, chat_history))
        explainer = shap.DeepExplainer(model, inputs['input_ids'])
        shap_values = explainer.shap_values(inputs['input_ids'])
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        shap_scores = {token: float(score) for token, score in zip(tokens, shap_values[0]) if token not in ['[CLS]', '[SEP]', '[PAD]']}
        st.session_state.last_shap_scores = shap_scores
        st.session_state.last_explainer = explainer
        try:
            c = conn.cursor()
            with conn:
                c.execute(
                    "INSERT INTO interactions (user_id, created_at, input_text, response, emotion, attention_scores, interaction_id) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    (user_id, created_at, text, response, emotion, json.dumps(shap_scores), interaction_id)
                )
                conn.commit()
            check_achievements(user_id, conn, "chat", created_at)
        except psycopg2.Error as e:
            logging.error(f"Database error saving interaction: {e.pgcode} - {e.pgerror}")
            conn.rollback()
        return response, emotion, shap_scores, created_at, explainer, interaction_id
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        emotion = detect_emotion_fallback(text)
        response = generate_motivational_response(text, emotion, chat_history, username, analyze_text(text, chat_history))
        try:
            c = conn.cursor()
            with conn:
                c.execute(
                    "INSERT INTO interactions (user_id, created_at, input_text, response, emotion, interaction_id) VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, created_at, text, response, emotion, interaction_id)
                )
                conn.commit()
            check_achievements(user_id, conn, "chat", created_at)
        except psycopg2.Error as e:
            logging.error(f"Database error saving interaction: {e.pgcode} - {e.pgerror}")
            conn.rollback()
        return response, emotion, shap_scores, created_at, explainer, interaction_id

# Get username
def get_username(user_id, conn):
    try:
        c = conn.cursor()
        c.execute("SELECT username FROM users WHERE user_id = %s", (user_id,))
        result = c.fetchone()
        return result[0] if result else "User"
    except psycopg2.Error as e:
        logging.error(f"Database error fetching username: {e.pgcode} - {e.pgerror}")
        return "User"
    except Exception as e:
        logging.error(f"Unexpected error fetching username: {e}")
        return "User"

# Password hashing
def hash_password(password):
    return int(hashlib.sha256(password.encode()).hexdigest(), 16) % 10000

# Get quote of the day
def get_quote_of_the_day():
    quotes = [
        "You are stronger than you know, and every step forward proves it.",
        "Embrace today with kindness and courage.",
        "Your mind is a garden; plant positive thoughts.",
        "Every small victory counts toward your growth.",
        "Breathe deeply, and let worries slip away."
    ]
    random.seed(datetime.now().date().toordinal())
    return random.choice(quotes)

# Home page
def home_page(conn):
    st.markdown("""
        <style>
        .main-title { font-size: 2.5em; color: #4B0082; text-align: center; }
        .welcome-text { font-size: 1.2em; text-align: center; margin-bottom: 20px; }
        .quote-text { font-size: 1.1em; font-style: italic; text-align: center; color: #333; margin: 20px 0; }
        .login-container { max-width: 400px; margin: auto; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .stButton>button { width: 100%; background-color: #4B0082; color: white; }
        .stTextInput>div>input { width: 100%; }
        </style>
        <div class="main-title">Welcome to MindMend AI</div>
        <div class="welcome-text">Your companion for mental wellness and emotional support.</div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="quote-text">"{get_quote_of_the_day()}"</div>', unsafe_allow_html=True)

    if not st.session_state.user_id:
        with st.container():
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.subheader("Login or Sign Up")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login"):
                    try:
                        c = conn.cursor()
                        c.execute("SELECT user_id, is_admin FROM users WHERE username = %s AND points = %s", 
                                 (username, hash_password(password)))
                        user = c.fetchone()
                        if user:
                            st.session_state.user_id = user[0]
                            st.session_state.is_admin = user[1]
                            c.execute("UPDATE users SET last_login = %s WHERE user_id = %s", 
                                     (datetime.now(), user[0]))
                            conn.commit()
                            st.success(f"Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    except psycopg2.Error as e:
                        logging.error(f"Database error during login: {e.pgcode} - {e.pgerror}")
                        st.error("Login failed. Please try again.")
                    except Exception as e:
                        logging.error(f"Unexpected error during login: {e}")
                        st.error("Login failed. Please try again.")
            with col2:
                if st.button("Sign Up"):
                    try:
                        c = conn.cursor()
                        user_id = f"user_{uuid.uuid4().int & (1<<64)-1}"
                        c.execute("SELECT 1 FROM users WHERE username = %s", (username,))
                        if c.fetchone():
                            st.error("Username already exists")
                        else:
                            with conn:
                                c.execute(
                                    "INSERT INTO users (user_id, username, points, is_admin, last_login) VALUES (%s, %s, %s, %s, %s)",
                                    (user_id, username, 0, False, datetime.now())
                                )
                                conn.commit()
                            st.session_state.user_id = user_id
                            st.session_state.is_admin = False
                            st.success(f"Welcome, {username}!")
                            st.rerun()
                    except psycopg2.Error as e:
                        logging.error(f"Database error during sign up: {e.pgcode} - {e.pgerror}")
                        st.error("Sign up failed. Please try again.")
                        conn.rollback()
                    except Exception as e:
                        logging.error(f"Unexpected error during sign up: {e}")
                        st.error("Sign up failed. Please try again.")
                        conn.rollback()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        username = get_username(st.session_state.user_id, conn)
        st.markdown(f'<div class="welcome-text">Hello, {username}! How can I support you today?</div>', unsafe_allow_html=True)
        try:
            c = conn.cursor()
            c.execute("SELECT points FROM users WHERE user_id = %s", (st.session_state.user_id,))
            points = c.fetchone()[0]
            st.markdown(f'<div class="welcome-text">Your Points: {points}</div>', unsafe_allow_html=True)
            c.execute("SELECT achievement, points, created_at FROM achievements WHERE user_id = %s ORDER BY created_at DESC", 
                     (st.session_state.user_id,))
            achievements = c.fetchall()
            if achievements:
                st.subheader("Recent Achievements")
                achievements_df = pd.DataFrame(achievements, columns=["Achievement", "Points", "Date"])
                st.write(achievements_df.head(5))
            else:
                st.info("No achievements yet. Try logging a mood or chatting to earn points!")
        except psycopg2.Error as e:
            logging.error(f"Database error fetching points/achievements: {e.pgcode} - {e.pgerror}")
            st.error("Failed to fetch points or achievements.")
        except Exception as e:
            logging.error(f"Unexpected error fetching points/achievements: {e}")
            st.error("Failed to fetch points or achievements.")

        st.markdown('<div class="mt-6">', unsafe_allow_html=True)
        st.subheader("Explore MindMend")
        cols = st.columns(3)
        with cols[0]:
            if st.button("Chat with Bot", key="home_chat"):
                st.session_state.page = "Chat with Bot"
                st.rerun()
        with cols[1]:
            if st.button("Track Mood", key="home_mood"):
                st.session_state.page = "Mood Tracker"
                st.rerun()
        with cols[2]:
            if st.button("Play Games", key="home_games"):
                st.session_state.page = "Games"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Chat with bot page
def chat_with_bot_page(conn, tokenizer, model):
    st.markdown("""
        <style>
        .chat-container { max-width: 800px; margin: auto; padding: 20px; }
        .user-message { background-color: #e6f3ff; padding: 10px; border-radius: 10px; margin: 5px 0; }
        .bot-message { background-color: #f0e6ff; padding: 10px; border-radius: 10px; margin: 5px 0; }
        .typing-indicator { display: flex; align-items: center; }
        .dot { height: 10px; width: 10px; background-color: #4B0082; border-radius: 50%; margin: 0 5px; animation: blink 1.4s infinite both; }
        .dot:nth-child(2) { animation-delay: 0.2s; }
        .dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes blink { 0% { opacity: 0.2; } 20% { opacity: 1; } 100% { opacity: 0.2; } }
        </style>
        <div class="chat-container">
    """, unsafe_allow_html=True)

    st.subheader("Chat with MindMend AI")
    st.markdown("Share your thoughts, and I'll provide supportive responses.")
    
    for message in st.session_state.chat_history:
        if message["type"] == "user":
            st.markdown(f'<div class="user-message"><strong>You</strong> ({message["created_at"]}): {message["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>MindMend AI</strong> ({message["created_at"]}): {message["text"]} ({message["emotion"].capitalize()})</div>', unsafe_allow_html=True)

    with st.form(key="chat_form"):
        user_input = st.text_area("What's on your mind?", height=100, key="chat_input")
        col1, col2 = st.columns([3, 1])
        with col2:
            submit_button = st.form_submit_button("Send")
        if submit_button and user_input.strip():
            typing_placeholder = st.empty()
            typing_placeholder.markdown(
                '<div class="typing-indicator"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>',
                unsafe_allow_html=True
            )
            time.sleep(1)
            try:
                response, emotion, shap_scores, created_at, explainer, interaction_id = generate_response(
                    user_input, tokenizer, model, st.session_state.user_id, conn, st.session_state.chat_history
                )
                st.session_state.chat_history.append({
                    "type": "user", "text": user_input, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "emotion": None
                })
                st.session_state.chat_history.append({
                    "type": "bot", "text": response, "emotion": emotion, "created_at": created_at,
                    "interaction_id": interaction_id
                })
            except Exception as e:
                logging.error(f"Error processing chat: {e}")
                st.error(f"An unexpected error occurred: {e}")
            typing_placeholder.empty()
            st.rerun()
        elif submit_button and not user_input.strip():
            st.warning("Please enter a message before sending.")

    if st.session_state.chat_history and st.session_state.chat_history[-1]["type"] == "bot":
        emotion = st.session_state.chat_history[-1]["emotion"]
        created_at = st.session_state.chat_history[-1]["created_at"]
        interaction_id = st.session_state.chat_history[-1]["interaction_id"]
        user_input = st.session_state.chat_history[-2]["text"] if len(st.session_state.chat_history) >= 2 else ""

        st.markdown(f"**Detected Emotion:** {emotion.capitalize()}")

        show_explanation = st.checkbox("Show explanation of this response", value=False, key=f"explain_{interaction_id}")
        if show_explanation:
            st.subheader("Why I Responded This Way")
            st.session_state.explanation_mode = st.radio(
                "Explanation Detail Level",
                ["Basic", "Detailed", "Technical"],
                index=["Basic", "Detailed", "Technical"].index(st.session_state.explanation_mode)
            )
            explanation = explain_response(
                user_input, emotion, st.session_state.last_shap_scores, tokenizer, st.session_state.chat_history
            )
            st.markdown(explanation, unsafe_allow_html=True)

            if st.session_state.last_shap_scores and model and tokenizer:
                st.markdown("**Visual Insights**")
                with st.expander("Explore Word Influence", expanded=True):
                    shap_data = pd.DataFrame({
                        "Word": list(st.session_state.last_shap_scores.keys()),
                        "Influence": list(st.session_state.last_shap_scores.values())
                    }).sort_values("Influence", ascending=False)
                    fig = px.bar(shap_data.head(10), x="Word", y="Influence", title="Words That Influenced My Response")
                    fig.update_layout(xaxis_title="Words", yaxis_title="Influence Score")
                    st.plotly_chart(fig, use_container_width=True)

                    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(
                        {k: v for k, v in st.session_state.last_shap_scores.items() if v > 0}
                    )
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)

        st.markdown("**Was this emotion correct?**")
        feedback = st.radio("", ["Yes", "No"], key=f"feedback_{interaction_id}", horizontal=True)
        feedback_comments = st.text_area("Additional comments (optional)", key=f"feedback_comments_{interaction_id}")
        if st.button("Submit Feedback", key=f"submit_feedback_{interaction_id}"):
            try:
                c = conn.cursor()
                with conn:
                    c.execute(
                        "UPDATE interactions SET feedback = %s WHERE interaction_id = %s",
                        (f"{feedback}: {feedback_comments}" if feedback_comments else feedback, interaction_id)
                    )
                    conn.commit()
                st.success("Thank you for your feedback!")
            except psycopg2.Error as e:
                logging.error(f"Database error saving feedback: {e.pgcode} - {e.pgerror}")
                st.error("Failed to save feedback.")
                conn.rollback()
            except Exception as e:
                logging.error(f"Unexpected error saving feedback: {e}")
                st.error("Failed to save feedback.")

    st.markdown('</div>', unsafe_allow_html=True)

# Mood tracker page
def mood_tracker_page(conn):
    st.markdown("""
        <style>
        .mood-table { width: 100%; border-collapse: collapse; }
        .mood-table th, .mood-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .mood-table th { background-color: #4B0082; color: white; }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Mood Tracker")
    st.markdown("Log your mood and reflect on your emotional journey.")

    with st.form(key="mood_form"):
        mood = st.selectbox("Select mood", ["Happy", "Sad", "Anxious", "Neutral"])
        notes = st.text_area("Add notes (optional)", height=100)
        submit_button = st.form_submit_button("Log Mood")
        if submit_button:
            try:
                c = conn.cursor()
                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                with conn:
                    c.execute(
                        "INSERT INTO mood_tracker (user_id, created_at, mood, notes) VALUES (%s, %s, %s, %s)",
                        (st.session_state.user_id, created_at, mood, notes)
                    )
                    conn.commit()
                check_achievements(st.session_state.user_id, conn, "mood_log", created_at)
                st.success("Mood logged successfully!")
                st.rerun()
            except psycopg2.Error as e:
                logging.error(f"Database error logging mood: {e.pgcode} - {e.pgerror}")
                st.error("Failed to log mood. Please try again.")
                conn.rollback()
            except Exception as e:
                logging.error(f"Unexpected error logging mood: {e}")
                st.error("Failed to log mood. Please try again.")
                conn.rollback()

    st.subheader("Mood History")
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    end_date = st.date_input("End Date", datetime.now())
    mood_filter = st.multiselect("Filter by Mood", ["Happy", "Sad", "Anxious", "Neutral"], default=["Happy", "Sad", "Anxious", "Neutral"])

    try:
        c = conn.cursor()
        c.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'mood_tracker')")
        if not c.fetchone()[0]:
            st.error("Mood tracker table not found.")
            logging.error("Mood tracker table not found in database")
            return
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d 23:59:59")
        logging.info(f"Mood query: user_id={st.session_state.user_id}, start={start_date_str}, end={end_date_str}, moods={mood_filter}")
        c.execute(
            """
            SELECT created_at, mood, notes 
            FROM mood_tracker 
            WHERE user_id = %s 
            AND created_at BETWEEN %s AND %s 
            AND mood = ANY(%s) 
            ORDER BY created_at DESC
            """,
            (st.session_state.user_id, start_date_str, end_date_str, mood_filter)
        )
        moods = c.fetchall()
        logging.info(f"Retrieved {len(moods)} mood entries")
        if moods:
            mood_df = pd.DataFrame(moods, columns=["Date", "Mood", "Notes"])
            st.markdown('<table class="mood-table">', unsafe_allow_html=True)
            st.write(mood_df)
            st.markdown('</table>', unsafe_allow_html=True)
            mood_counts = mood_df["Mood"].value_counts().reset_index()
            mood_counts.columns = ["Mood", "Count"]
            fig = px.pie(mood_counts, names="Mood", values="Count", title="Mood Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No mood data found for the selected period or filters.")
    except psycopg2.Error as e:
        logging.error(f"Database error fetching mood data: {e.pgcode} - {e.pgerror}")
        st.error(f"Database error fetching mood data: {e.pgerror}")
    except Exception as e:
        logging.error(f"Unexpected error fetching mood data: {e}")
        st.error(f"Unexpected error fetching mood data: {e}")

# Share moment page
def share_moment_page(conn):
    st.subheader("Share a Moment")
    st.markdown("Capture a moment that matters to you.")
    
    with st.form(key="moment_form"):
        moment = st.text_area("Describe your moment", height=100)
        category = st.selectbox("Category", ["Gratitude", "Challenge", "Joy", "Reflection"])
        submit_button = st.form_submit_button("Share Moment")
        if submit_button and moment.strip():
            try:
                c = conn.cursor()
                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                with conn:
                    c.execute(
                        "INSERT INTO moments (user_id, created_at, moment, category) VALUES (%s, %s, %s, %s)",
                        (st.session_state.user_id, created_at, moment, category)
                    )
                    conn.commit()
                check_achievements(st.session_state.user_id, conn, "moment_shared", created_at)
                st.success("Moment shared successfully!")
                st.rerun()
            except psycopg2.Error as e:
                logging.error(f"Database error sharing moment: {e.pgcode} - {e.pgerror}")
                st.error("Failed to share moment. Please try again.")
                conn.rollback()
            except Exception as e:
                logging.error(f"Unexpected error sharing moment: {e}")
                st.error("Failed to share moment. Please try again.")
                conn.rollback()
        elif submit_button and not moment.strip():
            st.warning("Please describe your moment before sharing.")

    st.subheader("Your Moments")
    try:
        c = conn.cursor()
        c.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'moments')")
        if not c.fetchone()[0]:
            st.error("Moments table not found.")
            logging.error("Moments table not found in database")
            return
        c.execute(
            "SELECT created_at, moment, category FROM moments WHERE user_id = %s ORDER BY created_at DESC LIMIT 10",
            (st.session_state.user_id,)
        )
        moments = c.fetchall()
        if moments:
            moments_df = pd.DataFrame(moments, columns=["Date", "Moment", "Category"])
            st.write(moments_df)
        else:
            st.info("No moments shared yet.")
    except psycopg2.Error as e:
        logging.error(f"Database error fetching moments: {e.pgcode} - {e.pgerror}")
        st.error(f"Failed to fetch moments: {e.pgerror}")
    except Exception as e:
        logging.error(f"Unexpected error fetching moments: {e}")
        st.error(f"Failed to fetch moments: {e}")

# Coping tools page
def coping_tools_page(conn):
    st.subheader("Coping Tools")
    st.markdown("Explore tools to manage stress and boost your well-being.")
    
    tool = st.selectbox("Choose a Tool", ["Quick Calm", "Gratitude List", "Vision Board"])
    if tool == "Quick Calm":
        st.markdown("**Quick Calm**: Follow the breathing guide below.")
        st.markdown("""
            <style>
            .breathing-circle {
                width: 100px;
                height: 100px;
                background-color: #4B0082;
                border-radius: 50%;
                margin: auto;
                animation: breathe 8s ease-in-out infinite;
            }
            @keyframes breathe {
                0% { transform: scale(1); }
                50% { transform: scale(1.5); }
                100% { transform: scale(1); }
            }
            .breathing-text {
                text-align: center;
                font-size: 1.2em;
                margin-top: 10px;
            }
            </style>
            <div class="breathing-circle"></div>
            <div class="breathing-text">Inhale (4s)... Hold (4s)... Exhale (4s)</div>
        """, unsafe_allow_html=True)
    elif tool == "Gratitude List":
        with st.form(key="gratitude_form"):
            gratitude = st.text_area("What are you grateful for today?", height=100)
            if st.form_submit_button("Add to List"):
                if gratitude.strip():
                    st.success("Added to your gratitude list!")
                else:
                    st.warning("Please enter something you're grateful for.")
    elif tool == "Vision Board":
        st.markdown("**Vision Board**: Create a collage of images and quotes to inspire you!")
        st.markdown("Add images (by description) and quotes to build your personal vision board.")

        if "vision_board" not in st.session_state:
            st.session_state.vision_board = []

        with st.form(key="vision_form"):
            image_desc = st.text_input("Describe an inspiring image (e.g., 'sunset', 'mountain')")
            quote = st.text_input("Add a motivational quote")
            submit = st.form_submit_button("Add to Vision Board")
            if submit and (image_desc.strip() or quote.strip()):
                st.session_state.vision_board.append({"image": image_desc, "quote": quote, "id": str(uuid.uuid4())})
                try:
                    c = conn.cursor()
                    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    with conn:
                        c.execute(
                            "INSERT INTO moments (user_id, created_at, moment, category) VALUES (%s, %s, %s, %s)",
                            (st.session_state.user_id, created_at, f"Vision Board: {image_desc} - {quote}", "Vision")
                        )
                        conn.commit()
                    check_achievements(st.session_state.user_id, conn, "moment_shared", created_at)
                    st.success("Added to your Vision Board!")
                    st.rerun()
                except psycopg2.Error as e:
                    logging.error(f"Database error saving vision board: {e.pgcode} - {e.pgerror}")
                    st.error("Failed to save to vision board.")
                    conn.rollback()

        if st.session_state.vision_board:
            st.subheader("Your Vision Board")
            cols = st.columns(3)
            for i, item in enumerate(st.session_state.vision_board):
                with cols[i % 3]:
                    st.markdown(f"""
                        <div style="background-color: #f0e6ff; padding: 10px; border-radius: 10px; margin: 5px;">
                            <strong>{item['image']}</strong><br>
                            <em>{item['quote']}</em>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button("Remove", key=f"remove_vision_{item['id']}"):
                        st.session_state.vision_board = [x for x in st.session_state.vision_board if x['id'] != item['id']]
                        st.rerun()
        else:
            st.info("Your vision board is empty. Add an image or quote to start!")

# Save game score
def save_score(game_name, score, conn):
    try:
        c = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        with conn:
            c.execute("INSERT INTO game_scores (user_id, game_name, score, created_at) VALUES (%s, %s, %s, %s)",
                      (st.session_state.user_id, game_name, score, created_at))
            conn.commit()
        check_achievements(st.session_state.user_id, conn, "game_completed", created_at)
        logging.info(f"Saved score {score} for {game_name} for user {st.session_state.user_id}")
    except psycopg2.Error as e:
        logging.error(f"Database error saving game score: {e.pgcode} - {e.pgerror}")
        st.error("Failed to save game score.")
        conn.rollback()
    except Exception as e:
        logging.error(f"Unexpected error saving game score: {e}")
        st.error("Failed to save game score.")

# Games page
def games_page(conn):
    st.markdown("""
        <style>
        .game-container { max-width: 800px; margin: auto; padding: 20px; }
        .game-card { width: 100px; height: 100px; background-color: #e6f3ff; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.5em; cursor: pointer; margin: 5px; }
        .game-card:hover { background-color: #d1e7ff; }
        .garden-container { background-color: #f0f0f0; padding: 20px; border-radius: 10px; }
        </style>
        <div class="game-container">
    """, unsafe_allow_html=True)

    st.subheader("Games")
    st.markdown("Play fun games to lift your mood and earn points!")
    
    if "game_state" not in st.session_state:
        st.session_state.game_state = {}
    selected_game = st.selectbox("Choose a Game", ["Mindful Mosaic", "Gratitude Garden", "Emotion Puzzle"])
    if st.button("Reset Game", key="reset_game"):
        st.session_state.game_state = {}
        st.session_state.last_game = None
        st.success("Game reset! Select a game to start fresh.")
        st.rerun()
    
    if "last_game" not in st.session_state or st.session_state.last_game != selected_game:
        st.session_state.game_state = {}
        st.session_state.last_game = selected_game
        logging.info(f"Reset game state for {selected_game}")

    with st.spinner("Loading game..."):
        try:
            if selected_game == "Mindful Mosaic":
                st.markdown("**Mindful Mosaic**: Match pairs of positive words to earn points. Focus on the words to feel calm.")
                if "mindful_mosaic" not in st.session_state.game_state:
                    words = ["Hope", "Hope", "Calm", "Calm", "Joy", "Joy", "Love", "Love"]
                    random.shuffle(words)
                    st.session_state.game_state["mindful_mosaic"] = {
                        "words": words,
                        "revealed": [False] * 8,
                        "matched": [False] * 8,
                        "first": None,
                        "score": 0,
                        "bg_color": "#E6F3FF"
                    }
                game_state = st.session_state.game_state["mindful_mosaic"]
                st.markdown(f'<div style="background-color: {game_state["bg_color"]}; padding: 10px; border-radius: 10px;">', unsafe_allow_html=True)
                cols = st.columns(4)
                for i in range(8):
                    with cols[i % 4]:
                        if game_state["matched"][i]:
                            st.markdown(f'<div class="game-card">{game_state["words"][i]}</div>', unsafe_allow_html=True)
                        elif game_state["revealed"][i]:
                            if st.button(game_state["words"][i], key=f"mosaic_revealed_{i}_{random.randint(0, 10000)}", help="Click to deselect"):
                                game_state["revealed"][i] = False
                                st.rerun()
                        else:
                            if st.button(" ", key=f"mosaic_hidden_{i}_{random.randint(0, 10000)}", help="Click to reveal"):
                                logging.info(f"Mindful Mosaic: Clicked tile {i}")
                                if game_state["first"] is None:
                                    game_state["first"] = i
                                    game_state["revealed"][i] = True
                                elif game_state["first"] != i:
                                    game_state["revealed"][i] = True
                                    if game_state["words"][game_state["first"]] == game_state["words"][i]:
                                        game_state["matched"][game_state["first"]] = True
                                        game_state["matched"][i] = True
                                        game_state["score"] += 10
                                        colors = ["#E6F3FF", "#D4F1F4", "#BCEAD5", "#AED8E6"]
                                        game_state["bg_color"] = colors[sum(game_state["matched"]) // 2 % len(colors)]
                                        game_state["first"] = None
                                        if all(game_state["matched"]):
                                            save_score("Mindful Mosaic", game_state["score"], conn)
                                            st.success(f"Game Over! Score: {game_state['score']}")
                                    else:
                                        time.sleep(0.5)
                                        game_state["revealed"][i] = False
                                        game_state["revealed"][game_state["first"]] = False
                                        game_state["first"] = None
                                    st.rerun()
                st.markdown(f"**Score**: {game_state['score']}")
                st.markdown('</div>', unsafe_allow_html=True)

            elif selected_game == "Gratitude Garden":
                st.markdown("**Gratitude Garden**: Plant seeds of gratitude to grow your garden. Each plant earns points!")
                if "gratitude_garden" not in st.session_state.game_state:
                    st.session_state.game_state["gratitude_garden"] = {
                        "plants": [],
                        "score": 0
                    }
                game_state = st.session_state.game_state["gratitude_garden"]
                with st.form(key="garden_form"):
                    gratitude = st.text_input("What are you grateful for?")
                    if st.form_submit_button("Plant Seed"):
                        if gratitude.strip():
                            game_state["plants"].append(gratitude)
                            game_state["score"] += 10
                            logging.info(f"Gratitude Garden: Planted '{gratitude}'")
                            st.success(f"Planted '{gratitude}'!")
                            st.rerun()
                        else:
                            st.warning("Please enter something you're grateful for.")
                st.markdown('<div class="garden-container">', unsafe_allow_html=True)
                if game_state["plants"]:
                    garden = "\n".join([f"🌱 {p}" for p in game_state["plants"]])
                    st.markdown(f"**Your Garden**:\n{garden}")
                else:
                    st.markdown("Your garden is empty. Plant a seed to start!")
                st.markdown(f"**Score**: {game_state['score']}")
                if st.button("Save Garden Score"):
                    save_score("Gratitude Garden", game_state["score"], conn)
                    st.success(f"Score saved: {game_state['score']}")
                st.markdown('</div>', unsafe_allow_html=True)

            elif selected_game == "Emotion Puzzle":
                st.markdown("**Emotion Puzzle**: Sort emotions into Positive or Negative categories. Learn about emotions as you play!")
                if "emotion_puzzle" not in st.session_state.game_state:
                    emotions = [
                        ("Joy", "Positive"), ("Sadness", "Negative"), ("Gratitude", "Positive"),
                        ("Anger", "Negative"), ("Hope", "Positive"), ("Fear", "Negative")
                    ]
                    random.shuffle(emotions)
                    st.session_state.game_state["emotion_puzzle"] = {
                        "emotions": emotions,
                        "current": 0,
                        "score": 0,
                        "feedback": ""
                    }
                game_state = st.session_state.game_state["emotion_puzzle"]
                if game_state["current"] < len(game_state["emotions"]):
                    emotion = game_state["emotions"][game_state["current"]][0]
                    st.markdown(f"**Sort this emotion**: {emotion}")
                    category = st.radio("Category", ["Positive", "Negative"], key=f"puzzle_{game_state['current']}")
                    if st.button("Submit", key=f"submit_puzzle_{game_state['current']}"):
                        logging.info(f"Emotion Puzzle: Sorted {emotion} as {category}")
                        correct_category = game_state["emotions"][game_state["current"]][1]
                        if category == correct_category:
                            game_state["score"] += 10
                            game_state["feedback"] = f"Correct! {emotion} is a {correct_category} emotion."
                            if emotion == "Joy":
                                game_state["feedback"] += " It’s associated with happiness and contentment."
                            elif emotion == "Sadness":
                                game_state["feedback"] += " It reflects loss or disappointment."
                            elif emotion == "Gratitude":
                                game_state["feedback"] += " It fosters appreciation and positivity."
                            elif emotion == "Anger":
                                game_state["feedback"] += " It can signal frustration or injustice."
                            elif emotion == "Hope":
                                game_state["feedback"] += " It inspires optimism and motivation."
                            elif emotion == "Fear":
                                game_state["feedback"] += " It’s a response to perceived threats."
                        else:
                            game_state["feedback"] = f"Oops! {emotion} is a {correct_category} emotion. Try the next one!"
                        game_state["current"] += 1
                        st.rerun()
                    if game_state["feedback"]:
                        st.markdown(game_state["feedback"])
                    st.markdown(f"**Score**: {game_state['score']}")
                else:
                    save_score("Emotion Puzzle", game_state["score"], conn)
                    st.success(f"Game Over! Score: {game_state['score']}")
                    if st.button("Play Again"):
                        st.session_state.game_state["emotion_puzzle"] = None
                        st.rerun()

        except Exception as e:
            logging.error(f"Error loading game {selected_game}: {e}")
            st.error(f"Failed to load {selected_game}. Please try another game or reset.")

    try:
        c = conn.cursor()
        c.execute("SELECT game_name, score, created_at FROM game_scores WHERE user_id = %s ORDER BY created_at DESC LIMIT 5",
                  (st.session_state.user_id,))
        scores = c.fetchall()
        if scores:
            scores_df = pd.DataFrame(scores, columns=["Game", "Score", "Date"])
            st.subheader("Recent Scores")
            st.write(scores_df)
        else:
            st.info("No game scores yet. Play a game to earn points!")
    except psycopg2.Error as e:
        logging.error(f"Database error fetching game scores: {e.pgcode} - {e.pgerror}")
        st.error("Failed to fetch game scores.")
    except Exception as e:
        logging.error(f"Unexpected error fetching game scores: {e}")
        st.error("Failed to fetch game scores.")

    st.markdown('</div>', unsafe_allow_html=True)

# Achievements page
def achievements_page(conn):
    st.markdown("""
        <style>
        .achievements-table { width: 100%; border-collapse: collapse; }
        .achievements-table th, .achievements-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .achievements-table th { background-color: #4B0082; color: white; }
        </style>
    """, unsafe_allow_html=True)

    st.subheader("Your Achievements")
    st.markdown("Track your progress and see the points you've earned!")

    try:
        c = conn.cursor()
        c.execute("SELECT points FROM users WHERE user_id = %s", (st.session_state.user_id,))
        total_points = c.fetchone()[0]
        st.markdown(f"**Total Points**: {total_points}")

        c.execute(
            "SELECT achievement, points, created_at FROM achievements WHERE user_id = %s ORDER BY created_at DESC",
            (st.session_state.user_id,)
        )
        achievements = c.fetchall()
        if achievements:
            achievements_df = pd.DataFrame(achievements, columns=["Achievement", "Points", "Date"])
            st.markdown('<table class="achievements-table">', unsafe_allow_html=True)
            st.write(achievements_df)
            st.markdown('</table>', unsafe_allow_html=True)
            
            achievements_df["Date"] = pd.to_datetime(achievements_df["Date"])
            points_over_time = achievements_df.groupby(achievements_df["Date"].dt.date)["Points"].sum().cumsum()
            fig = px.line(
                x=points_over_time.index, 
                y=points_over_time.values, 
                title="Points Earned Over Time",
                labels={"x": "Date", "y": "Total Points"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No achievements yet. Try logging a mood, chatting, or playing a game to earn points!")
    except psycopg2.Error as e:
        logging.error(f"Database error fetching achievements: {e.pgcode} - {e.pgerror}")
        st.error("Failed to fetch achievements.")
    except Exception as e:
        logging.error(f"Unexpected error fetching achievements: {e}")
        st.error("Failed to fetch achievements.")

# Admin dashboard page
def admin_dashboard_page(conn):
    st.markdown("""
        <style>
        .admin-container { max-width: 1000px; margin: auto; padding: 20px; }
        .metric-card { background-color: #f0f0f0; padding: 15px; border-radius: 10px; text-align: center; }
        </style>
        <div class="admin-container">
    """, unsafe_allow_html=True)

    st.subheader("Admin Dashboard")
    st.markdown("Monitor user activity and system performance.")
    
    try:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM interactions WHERE created_at >= %s", (datetime.now() - timedelta(days=7),))
        recent_interactions = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM mood_tracker WHERE created_at >= %s", (datetime.now() - timedelta(days=7),))
        recent_moods = c.fetchone()[0]
        cols = st.columns(3)
        with cols[0]:
            st.markdown(f'<div class="metric-card"><strong>Total Users</strong><br>{total_users}</div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="metric-card"><strong>Interactions (7 days)</strong><br>{recent_interactions}</div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f'<div class="metric-card"><strong>Moods Logged (7 days)</strong><br>{recent_moods}</div>', unsafe_allow_html=True)
    except psycopg2.Error as e:
        logging.error(f"Database error fetching admin metrics: {e.pgcode} - {e.pgerror}")
        st.error("Failed to fetch metrics.")
    except Exception as e:
        logging.error(f"Unexpected error fetching admin metrics: {e}")
        st.error("Failed to fetch metrics.")

    st.subheader("Recent User Activity")
    try:
        c = conn.cursor()
        c.execute("""
            SELECT u.username, i.input_text, i.response, i.created_at 
            FROM interactions i 
            JOIN users u ON i.user_id = u.user_id 
            ORDER BY i.created_at DESC LIMIT 10
        """)
        activities = c.fetchall()
        if activities:
            activity_df = pd.DataFrame(activities, columns=["Username", "Input", "Response", "Date"])
            st.write(activity_df)
        else:
            st.info("No recent activity.")
    except psycopg2.Error as e:
        logging.error(f"Database error fetching recent activity: {e.pgcode} - {e.pgerror}")
        st.error("Failed to fetch recent activity.")
    except Exception as e:
        logging.error(f"Unexpected error fetching recent activity: {e}")
        st.error("Failed to fetch recent activity.")

    st.markdown('<div class="mt-6">', unsafe_allow_html=True)
    st.subheader("Explore More")
    cols = st.columns(3)
    with cols[0]:
        if st.button("View User Activity", key="admin_activity"):
            st.session_state.page = "Home"
            st.rerun()
    with cols[1]:
        if st.button("Analyze Moods", key="admin_moods"):
            st.session_state.page = "Mood Tracker"
            st.rerun()
    with cols[2]:
        if st.button("Monitor Chats", key="admin_chats"):
            st.session_state.page = "Chat with Bot"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Main application logic
def main():
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_shap_scores" not in st.session_state:
        st.session_state.last_shap_scores = None
    if "last_explainer" not in st.session_state:
        st.session_state.last_explainer = None
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "explanation_mode" not in st.session_state:
        st.session_state.explanation_mode = "Basic"
    if "game_state" not in st.session_state:
        st.session_state.game_state = {}

    global conn
    conn = init_db()
    if conn is None:
        st.error("Failed to connect to database. Please try again later.")
        logging.error("Database connection failed")
        return

    tokenizer, model = load_model()
    if tokenizer is None or model is None:
        st.warning("Model loading failed. Some features may be limited.")

    st.sidebar.title("MindMend AI")
    st.sidebar.markdown("Navigate to different sections of your mental health companion.")
    page_options = ["Home", "Chat with Bot", "Mood Tracker", "Share a Moment", "Coping Tools", "Games", "Achievements"]
    if st.session_state.is_admin:
        page_options.append("Admin Dashboard")
    page = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.page))

    if page != st.session_state.page:
        st.session_state.page = page
        st.rerun()

    try:
        if page == "Home":
            home_page(conn)
        elif page == "Chat with Bot":
            if st.session_state.user_id:
                chat_with_bot_page(conn, tokenizer, model)
            else:
                st.error("Please log in to access the chat.")
                st.session_state.page = "Home"
                st.rerun()
        elif page == "Mood Tracker":
            if st.session_state.user_id:
                mood_tracker_page(conn)
            else:
                st.error("Please log in to track your mood.")
                st.session_state.page = "Home"
                st.rerun()
        elif page == "Share a Moment":
            if st.session_state.user_id:
                share_moment_page(conn)
            else:
                st.error("Please log in to share a moment.")
                st.session_state.page = "Home"
                st.rerun()
        elif page == "Coping Tools":
            if st.session_state.user_id:
                coping_tools_page(conn)
            else:
                st.error("Please log in to access coping tools.")
                st.session_state.page = "Home"
                st.rerun()
        elif page == "Games":
            if st.session_state.user_id:
                games_page(conn)
            else:
                st.error("Please log in to play games.")
                st.session_state.page = "Home"
                st.rerun()
        elif page == "Achievements":
            if st.session_state.user_id:
                achievements_page(conn)
            else:
                st.error("Please log in to view achievements.")
                st.session_state.page = "Home"
                st.rerun()
        elif page == "Admin Dashboard":
            if st.session_state.is_admin:
                admin_dashboard_page(conn)
            else:
                st.error("Access denied. Admin privileges required.")
                st.session_state.page = "Home"
                st.rerun()
    except Exception as e:
        logging.error(f"Error rendering page {page}: {e}")
        st.error(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            try:
                conn.close()
                logging.info("Database connection closed")
            except psycopg2.Error as e:
                logging.error(f"Error closing database connection: {e.pgcode} - {e.pgerror}")

if __name__ == "__main__":
    main()
