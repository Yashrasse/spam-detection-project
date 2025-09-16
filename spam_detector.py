import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


# background image
import base64

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_background("photo1.jpg")


# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

download_nltk_data()

# Set page config
st.set_page_config(
    page_title="Spam-Detector",
    page_icon="🚫",
    layout="wide",
    initial_sidebar_state="expanded"
)
from streamlit_lottie import st_lottie


# Define your custom CSS with the animation
animation_css = """
<style>

.main-header{
    padding: 15px;
    border-radius: 10px;
    animation: rotate_gradient 4s linear infinite;
	background: linear-gradient(-40deg,#ff0000, #000000, #000000);
	background-size: 200% 100%;
	animation: gradient 10s ease infinite;
	height: auto;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 30px;
}

@keyframes gradient {
	0% {
		background-position: 0% 50%;
	}
	50% {
		background-position: 100% 50%;
	}
	100% {
		background-position: 0% 50%;
	}
}

</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(animation_css, unsafe_allow_html=True)
# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .spam-prediction {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .ham-prediction {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
# st.markdown("""
# <div class="main-header">
#     <h1>🚫Spam Detector</h1>
#     <p>Email & SMS Spam Message Detection using Machine Learning</p>
# </div>
# """, unsafe_allow_html=True)



# Inject CSS to make the sidebar background transparent
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("🔧 Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    [  "🔍 Make Prediction","🛡️ Security Tips","📋 Project Info"]
)

# Text preprocessing function
@st.cache_data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        text = ' '.join([word for word in words if word not in stop_words])
    except:
        pass
    
    # Stemming
    try:
        stemmer = PorterStemmer()
        words = text.split()
        text = ' '.join([stemmer.stem(word) for word in words])
    except:
        pass
    
    return text

@st.cache_data
def generate_sample_data():
    # Load CSV and select only relevant columns to avoid unnamed NaN columns
    df = pd.read_csv("spam.csv")[['label', 'message']]
    
    # Drop rows where 'message' is NaN (if any)
    df = df.dropna(subset=['message'])
    
    # Ensure labels are lowercase for consistency (e.g., 'ham' not 'Ham')
    df['label'] = df['label'].str.lower()
    
    # If you need to add random duplicates/variations
    spam_messages = df[df['label'] == 'spam']['message'].tolist()
    ham_messages = df[df['label'] == 'ham']['message'].tolist()
    messages = spam_messages + ham_messages
    labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
    
    additional_data = []
    for i in range(30):  # Same as original
        idx = np.random.randint(0, len(messages))
        additional_data.append({
            'message': messages[idx],
            'label': labels[idx]
        })
    
    additional_df = pd.DataFrame(additional_data)
    df = pd.concat([df, additional_df], ignore_index=True)
    
    # Ensure each class has at least 2 samples to allow stratified split
    class_counts = df['label'].value_counts()
    for label in class_counts.index:
        if class_counts[label] < 2:
            min_samples = df[df['label'] == label]
            df = pd.concat([df, min_samples], ignore_index=True)  # Duplicate to make ≥2
    
    return df




#Generate sample dataset
# @st.cache_data
# def generate_sample_data():
#     # Sample spam and ham messages
#     spam_messages = [
#         "Congratulations! You've won $1000! Click here to claim now!",
#         "URGENT: Your account will be closed. Call 123-456-7890 immediately!",
#         "Free iPhone! Limited time offer! Text STOP to unsubscribe.",
#         "You've been selected for a free cruise! Call now!",
#         "Make money fast! Work from home opportunity!",
#         "WINNER! You've won a lottery! Send your bank details!",
#         "Exclusive deal! 90% off everything! Buy now!",
#         "Your credit card has been charged $500. Call to dispute!",
#         "Free gift card worth $100! Click link to redeem!",
#         "Lose weight fast! Miracle pill available now!",
#         "Nigerian prince needs help transferring money. Reward offered!",
#         "Your subscription expires today! Renew now for discount!",
#         "Hot singles in your area! Meet tonight!",
#         "You owe money! Pay immediately or face legal action!",
#         "Free vacation! All expenses paid! Call now!",
#         "Medicine at 70% discount! No prescription needed!",
#         "Make $5000 per week working from home!",
#         "Your package is delayed. Click to track!",
#         "Tax refund available! Claim now before it expires!",
#         "Free ringtones! Download now!",
#         "Click on this link to get free gifts",
#         "Make 5000rs per week working from home!",
#         "Hey, it's I'm in a meeting now and need your help with something urgent. Can you transfer $5,000 to this account ASAP? I'll explain everything later. Please keep this confidential.",
#         "CONGRATULATIONS! You have been selected to receive a cash prize of $10000! Reply YES to claim",
#         "URGENT! Your bank account has been compromised. Click here to secure it immediately",
#         "LIMITED TIME: Get 99% discount on all products! Hurry up!",
#         "You are a WINNER! Claim your prize money now before it expires!",
#         "ALERT: Suspicious activity detected. Verify your account details immediately",
#         "Free money! No strings attached! Click to get instant cash",
#         "Your loan has been approved for $50000! Apply now!"
#     ]
    
#     ham_messages = [
#         "Hey, are we still meeting for lunch tomorrow?",
#         "Thanks for the great presentation today!",
#         "Can you send me the project files by Friday?",
#         "Don't forget about the team meeting at 2 PM.",
#         "Happy birthday! Hope you have a wonderful day!",
#         "The weather is perfect for a walk today.",
#         "I'll pick up groceries on my way home.",
#         "Great job on the presentation! Well done!",
#         "See you at the conference next week.",
#         "Please review the attached document when you get a chance.",
#         "The movie was amazing! You should watch it.",
#         "Thanks for helping me with the project.",
#         "Let me know when you're free to chat.",
#         "The restaurant was booked, let's try somewhere else.",
#         "I'm running 10 minutes late for our meeting.",
#         "Could you please check your email?",
#         "The package arrived safely, thank you!",
#         "Looking forward to our vacation next month!",
#         "The new policy takes effect next Monday.",
#         "Have a safe trip! Text me when you arrive.",
#         "Hello, how are you doing today?",
#         "Hi there, just checking in on you",
#         "Good morning! Hope you have a great day",
#         "Thanks for your help with the assignment",
#         "The meeting has been rescheduled to 3 PM",
#         "Can you please bring the documents tomorrow?",
#         "I enjoyed our conversation yesterday",
#         "The event was really successful, well organized",
#         "Please let me know if you need any assistance",
#         "Have a wonderful weekend with your family"
#     ]
    
# Create DataFrame
    messages = spam_messages + ham_messages
    labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
    
    df = pd.DataFrame({
        'message': messages,
        'label': labels
    })
    
    # Add some random duplicates and variations
    additional_data = []
    for i in range(30):
        idx = np.random.randint(0, len(messages))
        additional_data.append({
            'message': messages[idx],
            'label': labels[idx]
        })
    
    additional_df = pd.DataFrame(additional_data)
    df = pd.concat([df, additional_df], ignore_index=True)
    
    return df

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = generate_sample_data()
    df['processed_message'] = df['message'].apply(preprocess_text)
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    return df

# Train models
# @st.cache_resource
# def train_models(X_train, X_test, y_train, y_test):
#     models = {
#         'Naive Bayes': Pipeline([
#             ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english',ngram_range=(1, 2))),
#             ('classifier', MultinomialNB(alpha=1.0))
#         ]),
#         'Logistic Regression': Pipeline([
#             ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english',ngram_range=(1, 2))),
#             ('classifier', LogisticRegression(random_state=42,max_iter=1000))
#         ]),
#         'Random Forest': Pipeline([
#             ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
#             ('classifier', RandomForestClassifier(n_estimators=100, random_state=50,))
#         ])
#     }
    
#     trained_models = {}
#     model_scores = {}
    
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
        
#         trained_models[name] = model
#         model_scores[name] = {
#             'accuracy': accuracy,
#             'predictions': y_pred
#         }
    
#     return trained_models, model_scores

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=2000,  # Increased from 1000
                stop_words='english',
                ngram_range=(1, 3),  # Include trigrams for better context
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.9,  # Ignore terms that appear in more than 90% of documents
                sublinear_tf=True  # Apply sublinear tf scaling
            )),
            ('classifier', MultinomialNB(alpha=0.1))  # Reduced alpha for less smoothing
        ]),
        
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=2000,  # Increased iterations
                C=10.0,  # Regularization strength (higher = less regularization)
                class_weight='balanced',  # Handle class imbalance
                solver='liblinear'  # Better for smaller datasets
            ))
        ]),
        
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 2),  # Reduced for Random Forest to avoid overfitting
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                random_state=42,
                max_depth=15,  # Increased from 10
                min_samples_split=5,  # Prevent overfitting
                min_samples_leaf=2,  # Prevent overfitting
                class_weight='balanced',  # Handle class imbalance
                bootstrap=True
            ))
        ]),
        
        # Added: Support Vector Machine for better performance
        'SVM': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )),
            ('classifier', SVC(
                random_state=42,
                kernel='linear',  # Linear kernel works well for text
                C=1.0,
                class_weight='balanced',
                probability=True  # Enable probability predictions
            ))
        ]),
        
        # Added: Gradient Boosting for ensemble learning
        'Gradient Boosting': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.9,
                sublinear_tf=True
            )),
            ('classifier', GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=7,
                subsample=0.8
            ))
        ])
    }
    
    trained_models = {}
    model_scores = {}
    
    for name, model in models.items():
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate multiple metrics for better evaluation
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate F1 score (harmonic mean of precision and recall)
            from sklearn.metrics import f1_score, precision_score, recall_score
            f1 = f1_score(y_test, y_pred, pos_label='spam', average='binary')
            precision = precision_score(y_test, y_pred, pos_label='spam', average='binary')
            recall = recall_score(y_test, y_pred, pos_label='spam', average='binary')
            
            # Calculate ROC AUC score
            from sklearn.metrics import roc_auc_score
            # Get probability for spam class
            spam_idx = list(model.classes_).index('spam') if 'spam' in model.classes_ else 1
            y_test_binary = (y_test == 'spam').astype(int)
            auc = roc_auc_score(y_test_binary, y_pred_proba[:, spam_idx])
            
            # Composite score combining multiple metrics
            composite_score = (accuracy * 0.3) + (f1 * 0.4) + (auc * 0.3)
            
            trained_models[name] = model
            model_scores[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'composite_score': composite_score,  # Use this for model selection
                'predictions': y_pred
            }
            
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    return trained_models, model_scores






# Main app logic


if app_mode == "🔍 Make Prediction":
    st.markdown("""
<div class="main-header">
    <h1>🚫Spam Detector</h1>
    <p>Email & SMS Spam Message Detection using Machine Learning</p>
</div>
""", unsafe_allow_html=True)
    st.header("📝 Enter Message to Analyze")
    
   # st.subheader("📝 Enter Message to Analyze")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["✍️ Type Message", "📁 Upload File"])
    
    if input_method == "✍️ Type Message":
        message_input = st.text_area("Enter your message:", 
                                   placeholder="Type or paste your message here...",
                                   height=150)
        
        if st.button("🔍 Predict", type="primary"):
            if not message_input.strip():
                st.error("❌ Please enter a valid message to analyze.")
            else:
                # Auto-train models if not already trained
                needs_retraining = ('trained_models' not in st.session_state or 
                                'model_scores' not in st.session_state or
                                # Check if old format (missing new metrics)
                                any('f1_score' not in score_data for score_data in st.session_state.get('model_scores', {}).values()))
                if needs_retraining:
                    # Clear old session state to force retraining
                    if 'trained_models' in st.session_state:
                        del st.session_state.trained_models
                    if 'model_scores' in st.session_state:
                        del st.session_state.model_scores

                with st.spinner("Loading data and training models... Please wait!"):
                    # Load and preprocess data
                    df = load_and_preprocess_data()
                    
                    # Default training configuration
                    test_size = 0.3
                    random_state = 42
                    use_processed = True
                    
                    # Prepare data
                    X = df['processed_message'] if use_processed else df['message']
                    y = df['label']
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )
                    
                    # Train models
                    trained_models, model_scores = train_models(X_train, X_test, y_train, y_test)
                    
                    # Store in session state
                    st.session_state.trained_models = trained_models
                    st.session_state.model_scores = model_scores
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    #st.success("✅ Models trained successfully!")
                
                # Now make predictions with trained models
                processed_message = preprocess_text(message_input)
                
                # Get all model predictions and find the best one
                model_results = {}
                all_predictions = {}
                
                with st.spinner("Running all models and selecting the best..."):
                    for model_name, model in st.session_state.trained_models.items():
                        try:
                            prediction = model.predict([processed_message])[0]
                            probability = model.predict_proba([processed_message])[0]
                            
                            # Get probabilities
                            prob_spam = probability[1] if model.classes_[1] == 'spam' else probability[0]
                            prob_ham = probability[0] if model.classes_[0] == 'ham' else probability[1]
                            
                            model_metrics = st.session_state.model_scores[model_name]
            
                            model_results[model_name] = {
                                'prediction': prediction,
                                'prob_spam': prob_spam,
                                'prob_ham': prob_ham,
                                'accuracy': model_metrics.get('accuracy', 0),
                                'f1_score': model_metrics.get('f1_score', 0),
                                'precision': model_metrics.get('precision', 0),
                                'recall': model_metrics.get('recall', 0),
                                'auc': model_metrics.get('auc', 0),
                                'composite_score': model_metrics.get('composite_score', model_metrics.get('accuracy', 0)),
                                'confidence': prob_spam if prediction == 'spam' else prob_ham
                            }
                        except Exception as e:
                            st.error(f"Error with model {model_name}: {str(e)}")
                            continue
                        all_predictions[model_name] = {
                            'prediction': prediction,
                            'confidence': prob_spam if prediction == 'spam' else prob_ham
                        }
                    
                    # Find the best model using composite score (or accuracy as fallback)
                    if model_results:
                        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['composite_score'])
                        best_result = model_results[best_model_name]
                        
                        # Display BEST PREDICTION RESULT prominently
                        st.subheader("📋 PREDICTION RESULT")
                        
                        if best_result['prediction'] == 'spam':
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #000000, #FF0000);
                                padding: 25px;
                                border-radius: 15px;
                                border-left: 3px solid #FF0000;
                                margin: 20px 0;
                                box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3);
                                color: white;
                                text-align: center;
                            ">
                                <h2 style="margin: 0; font-size: 2.2em;">🚨 SPAM DETECTED: Fraud Message </h2>
                                <p style="font-size: 1.1em; margin-top: 10px;">Best Model: {best_model_name} | Confidence: {best_result['confidence']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="
                                background: linear-gradient(135deg, #000000, #00FF57);
                                padding: 25px;
                                border-radius: 15px;
                                border-left: 3px solid #00FF57;
                                margin: 20px 0;
                                box-shadow: 0 8px 25px rgba(46, 213, 115, 0.3);
                                color: white;
                                text-align: center;
                            ">
                                <h2 style="margin: 0; font-size: 2.2em;">✅ NOT A SPAM: It's a Real message</h2>
                            <p style="font-size: 1.1em; margin-top: 10px;">Best Model: {best_model_name} | Confidence: {best_result['confidence']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show all model results in expandable section
                        with st.expander("📊 View All Model Results"):
                            st.subheader("🎯 All Model Predictions")
                            
                            # Sort models by accuracy (descending)
                            sorted_models = sorted(model_results.items(), key=lambda x: x[1]['composite_score'], reverse=True)
                            
                            for i, (model_name, result) in enumerate(sorted_models):
                                rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."

                                metrics_text = f"<strong>Score:</strong> {result['composite_score']:.1%} | <strong>Confidence:</strong> {result['confidence']:.1%}"
                                # Add additional metrics if available

                                if result['f1_score'] > 0:
                                    metrics_text += f"<br><strong>Accuracy:</strong> {result['accuracy']:.1%} | <strong>F1:</strong> {result['f1_score']:.1%} | <strong>AUC:</strong> {result['auc']:.1%}"
                                
                                if result['prediction'] == 'spam':
                                    st.markdown(f"""
                                    <div style="
                                        background: {'linear-gradient(135deg, #000000, #ff0000)' if i == 0 else "#FBBDC6"};
                                        padding: 15px;
                                        border-radius: 10px;
                                        border-left: 3px solid #fc0303;
                                        margin: 10px 0;
                                        {'color: white;' if i == 0 else 'color: #333;'}
                                    ">
                                        <h4>{rank_emoji} {model_name}: SPAM DETECTED</h4>
                                        <p>{metrics_text}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div style="
                                        background: {'linear-gradient(135deg, #000000, #00ff00)' if i == 0 else "#EBFEEB"};
                                        padding: 15px;
                                        border-radius: 10px;
                                        border-left: 3px solid #0bfc03;
                                        margin: 10px 0;
                                        {'color: white;' if i == 0 else 'color: #333;'}
                                    ">
                                        <h4>{rank_emoji} {model_name}: NOT A SPAM</h4>
                                        <p>{metrics_text}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Message analysis
                        st.subheader("📊 Message Analysis")
                        
                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            st.metric("📝 Message Length", len(message_input))

                        with col2:
                            st.metric("📢 Word Count", len(message_input.split()))

                        with col3:
                            char_density = len(message_input.replace(' ', ''))/len(message_input) if message_input else 0
                            st.metric("🔤 Character Density", f"{char_density:.2f}")

                        with col4:
                            # Count of models that detected spam
                            spam_detections = sum(1 for result in model_results.values() if result['prediction'] == 'spam')
                            st.metric("🤖 Models Detecting Spam", f"{spam_detections}/{len(model_results)}")

                        with col5:
                            # Average confidence across all models
                            avg_confidence = sum(result['confidence'] for result in model_results.values()) / len(model_results)
                            st.metric("📊 Avg Confidence", f"{avg_confidence:.1%}")

                    else:
                        st.error("❌ Error: No models available for prediction. Please try again.")
                # Show processed version
                with st.expander("🔍 View Processed Message"):
                    st.write("**Original:**", message_input)
                    st.write("**Processed:**", processed_message)
        # Machine Learning Models Section
    #with st.expander("Used -🤖 Machine Learning Models "):
    else:  # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                
                st.write("📄 **File Preview:**")
                st.dataframe(df_upload.head())
                
                # Select message column
                message_column = st.selectbox("Select message column:", df_upload.columns)
                
                if st.button("🔍 Predict All", type="primary"):
                    # Auto-train models if not already trained
                    if 'trained_models' not in st.session_state:
                        st.info("🤖 Training models automatically for batch prediction...")
                        
                        with st.spinner("Loading data and training models... Please wait!"):
                            # Load and preprocess data
                            df = load_and_preprocess_data()
                            
                            # Default training configuration
                            test_size = 0.2
                            random_state = 42
                            use_processed = True
                            
                            # Prepare data
                            X = df['processed_message'] if use_processed else df['message']
                            y = df['label']
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state, stratify=y
                            )
                            
                            # Train models
                            trained_models, model_scores = train_models(X_train, X_test, y_train, y_test)
                            
                            # Store in session state
                            st.session_state.trained_models = trained_models
                            st.session_state.model_scores = model_scores
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            
                            st.success("✅ Models trained successfully!")
                    
                    with st.spinner("Making predictions with the best model..."):
                        # Process messages
                        messages = df_upload[message_column].astype(str)
                        processed_messages = messages.apply(preprocess_text)
                        
                        # Get best model (highest accuracy)
                        best_model_name = max(st.session_state.model_scores.keys(), 
                                            key=lambda x: st.session_state.model_scores[x]['accuracy'])
                        best_model = st.session_state.trained_models[best_model_name]
                        best_accuracy = st.session_state.model_scores[best_model_name]['accuracy']
                        
                        # Make predictions
                        predictions = best_model.predict(processed_messages)
                        probabilities = best_model.predict_proba(processed_messages)
                        
                        # Add results to dataframe
                        df_upload['prediction'] = predictions
                        df_upload['spam_probability'] = probabilities[:, 1] if best_model.classes_[1] == 'spam' else probabilities[:, 0]
                        df_upload['model_used'] = best_model_name
                        
                        # Display best model info prominently
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #3742fa, #5352ed);
                            padding: 20px;
                            border-radius: 10px;
                            color: white;
                            text-align: center;
                            margin: 20px 0;
                        ">
                            <h3>🏆 PREDICTIONS COMPLETED</h3>
                            <h4>Best Model Used: {best_model_name}</h4>
                            <p>Model Accuracy: {best_accuracy:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show results
                        st.subheader("📊 Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        spam_count = sum(df_upload['prediction'] == 'spam')
                        ham_count = sum(df_upload['prediction'] == 'ham')
                        total_messages = len(df_upload)
                        
                        with col1:
                            st.metric("🚨 Spam Detected", spam_count, f"{(spam_count/total_messages)*100:.1f}%")
                        
                        with col2:
                            st.metric("✅ Legitimate Messages", ham_count, f"{(ham_count/total_messages)*100:.1f}%")
                        
                        with col3:
                            st.metric("📋 Total Processed", total_messages)
                        
                        # Download results
                        csv = df_upload.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Results as CSV",
                            data=csv,
                            file_name=f"spam_detection_results_{best_model_name.lower().replace(' ', '_')}.csv",
                            mime="text/csv"
                        )
                        
                        # Show sample results
                        st.subheader("📋 Sample Results")
                        display_df = df_upload[[message_column, 'prediction', 'spam_probability', 'model_used']].head(10)
                        display_df['spam_probability'] = display_df['spam_probability'].round(3)
                        st.dataframe(display_df)
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}") 
            
        

elif app_mode == "🛡️ Security Tips":
    st.header("🛡️ Spam Protection Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 Common Spam Indicators")
        
        with st.expander("💰 Money-Related Red Flags"):
            st.write("• You won money/prizes!")
            st.write("• Send money now")
            st.write("• Get rich quick schemes")
            st.write("• Free money offers")
        
        with st.expander("⏰ Urgency Tactics"):
            st.write("• Act now or lose forever")
            st.write("• Limited time only")
            st.write("• Expires today")
            st.write("• Immediate action required")
        
        with st.expander("📱 Suspicious Contacts"):
            st.write("• Unknown phone numbers")
            st.write("• Fake email addresses")
            st.write("• Generic greetings")
            st.write("• Poor grammar/spelling")
    
    with col2:
        st.subheader("🛡️ Protection Methods")
        
        with st.expander("📧 Email Security"):
            st.write("• Enable spam filters")
            st.write("• Don't click suspicious links")
            st.write("• Verify sender identity")
            st.write("• Report spam messages")
        
        with st.expander("📱 SMS Security"):
            st.write("• Block unknown numbers")
            st.write("• Don't reply to spam")
            st.write("• Use carrier spam protection")
            st.write("• Be cautious with links")
        
        with st.expander("🔐 General Tips"):
            st.write("• Keep personal info private")
            st.write("• Use strong passwords")
            st.write("• Enable 2-factor authentication")
            st.write("• Stay updated on scam trends")
    
    st.warning("⚠️ **Important:** This tool is for educational purposes. Always verify suspicious messages through official channels!")

elif app_mode == "📋 Project Info":
            st.header("📋 Project Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #000000, #667eea);
                    padding: 25px;
                    border-radius: 12px;
                    border-left: 5px solid #667eea;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                ">
                    <h3 style="color: #667eea; margin-top: 0;">👨‍🎓 Students Information</h3>
                    <p><strong>Students Name:</strong> Yash & Karan</p>
                    <p><strong>Class:</strong> B.Sc. (Data Science Sem V)</p>
                    <p><strong>Project Title:</strong> Spam Detector - Email & SMS Spam Detection System</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #000000, #5403a6);
                    padding: 25px;
                    border-radius: 12px;
                    border-left: 5px solid #5403a6;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                ">
                    <h3 style="color: #764ba2; margin-top: 0;">🛠️ Technical Stack</h3>
                    <p><strong>Language:</strong> Python</p>
                    <p><strong>Platform:</strong> VS Code</p>
                    <p><strong>Data:</strong> SMS Spam Collection Dataset from - Kaggle</p>
                </div>
                """, unsafe_allow_html=True)
                
                
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #000000, #a902bd);
                    padding: 25px;
                    border-radius: 12px;
                    border-left: 5px solid #a902bd;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                ">
                    <h3 style="color: #f093fb; margin-top: 0;">📚 Libraries & Packages</h3>
                    <ul style="line-height: 1.8;">
                        <li><strong>Streamlit</strong> - Web Interface</li>
                        <li><strong>Pandas</strong> - Data Manipulation</li>
                        <li><strong>NumPy</strong> - Numerical Computing</li>
                        <li><strong>Scikit-learn</strong> - Machine Learning</li>
                        <li><strong>NLTK</strong> - Natural Language Processing</li>
                        <li><strong>Matplotlib & Seaborn</strong> - Data Visualization</li>
                        <li><strong>Plotly</strong> - Interactive Plots</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #000000, #f56e73);
                    padding: 25px;
                    border-radius: 12px;
                    border-left: 5px solid #f56e73;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                ">
                    <h3 style="color: #ff9a9e; margin-top: 0;">🧠 NLP Techniques</h3>
                    <ul style="line-height: 1.8;">
                        <li><strong>Text Preprocessing</strong></li>
                        <li><strong>Tokenization</strong></li>
                        <li><strong>Stop Words Removal</strong></li>
                        <li><strong>Stemming</strong></li>
                        <li><strong>TF-IDF Vectorization</strong></li>
                        <li><strong>Feature Extraction</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
                # System Architecture
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #000000, #635f5f);
                    padding: 25px;
                    border-radius: 12px;
                    border-left: 5px solid #635f5f;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                "><h3>🏗️ System Architecture</h3>
                </div>
                """, unsafe_allow_html=True)

            # Add a brief intro for context
                st.markdown("This diagram outlines the end-to-end pipeline for our Spam Detector model.")

            # Use columns or custom CSS for a more diagram-like feel, but keep it simple
                with st.expander("Step 1: Raw Text Input"):
                    st.markdown("""
                - User provides raw email/SMS text.
                - Example: "Congratulations! You've won a free iPhone. Click here to claim."
                """)

                    st.markdown("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")  # Shorten arrows for cleaner look

                with st.expander("Step 2: Text Preprocessing (6 steps)"):
                    st.markdown("""
                1. Lowercase conversion
                2. Remove punctuation
                3. Tokenization
                4. Stopword removal
                5. Stemming/Lemmatization
                6. Handle special characters/emojis
                """)

                    st.markdown("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")

                with st.expander("Step 3: TF-IDF Vectorization (max 1000 features)"):
                    st.markdown("""
                - Convert text to numerical features using TF-IDF.
                - Limit to top 1000 features for efficiency.
                """)

                    st.markdown("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")

                with st.expander("Step 4: ML Model Pipeline (5 models)"):
                    st.markdown("""
                - Train/evaluate multiple models: e.g., 1.Multinomial Naive Bayes, 2.Logistic Regression, 3.Random Forest, 4.Support Vector Machine, 5.Gradient Boosting.
                - Use a pipeline for streamlined training.
                """)
                    st.markdown("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")

                with st.expander("Step 5: Best Model Selection"):
                    st.markdown("""
                - Evaluate models using accuracy, F1-score, etc.
                - Select the best based on cross-validation.
                """)

                    st.markdown("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")

                with st.expander("Step 6: Prediction Result"):
                        st.markdown("We get result: SPAM or NOT SPAM ")
            
                    
                        st.error("SPAM")
                        st.write("OR")
                        st.success("NOT SPAM")
            
            # System Architecture
            # st.subheader("🏗️ System Architecture")

            # with st.expander("Step 1. Raw Text Input"):
            #      st.markdown("""↓↓↓↓↓↓↓↓↓↓↓↓↓↓""")
            # with st.expander("Step 2.ext Preprocessing (6 steps)"):
            #      st.markdown("""↓↓↓↓↓↓↓↓↓↓↓↓↓↓""")
            # with st.expander("Step 3.TF-IDF Vectorization (max 1000 features)"):
            #      st.markdown("""↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓""")
            # with st.expander("Step 4.ML Model Pipeline (3 models)"):
            #      st.markdown("""↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓""")
            # with st.expander("Step 5.Best Model Selection"):
            #      st.markdown("""↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓""")
            # with st.expander("Step 6.Prediction Result"):
            #      st.markdown("""  We get result SPAM or NOT SPAM """)
            #      st.success("NOT SPAM")
            #      st.write("OR")
            #      st.error("SPAM")

            st.subheader("🤖 Machine Learning Models :")
            
            col1, col2 = st.columns(2)

            with col1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #000000, #0d47a1); padding: 15px; border-radius: 10px; border-left: 5px solid #2196f3;">
                        <h4>📊 Multinomial Naive Bayes</h4>
                        <p><strong>Best for:</strong> Text classification</p>
                        <p><strong>Strengths:</strong> Fast, works with small datasets</p>
                        <p><strong>Use case:</strong> Spam detection baseline</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("  ")
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #000000, #1b5e20); padding: 15px; border-radius: 10px; border-left: 5px solid #4caf50;">
                        <h4>🌳 Random Forest</h4>
                        <p><strong>Best for:</strong> Complex patterns</p>
                        <p><strong>Strengths:</strong> Handles overfitting</p>
                        <p><strong>Use case:</strong> Ensemble learning</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("  ")
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #000000, #e65100); padding: 15px; border-radius: 10px; border-left: 5px solid #ff9800;">
                        <h4>🚀 Gradient Boosting</h4>
                        <p><strong>Best for:</strong> Sequential learning</p>
                        <p><strong>Strengths:</strong> High accuracy</p>
                        <p><strong>Use case:</strong> Advanced ensemble</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #000000, #4a148c); padding: 15px; border-radius: 10px; border-left: 5px solid #8b42e3;">
                        <h4>📈 Logistic Regression</h4>
                        <p><strong>Best for:</strong> Binary classification</p>
                        <p><strong>Strengths:</strong> Interpretable, probabilistic</p>
                        <p><strong>Use case:</strong> Linear decision boundaries</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("  ")
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #000000, #b71c1c); padding: 15px; border-radius: 10px; border-left: 5px solid #f44336;">
                        <h4>⚡ Support Vector Machine</h4>
                        <p><strong>Best for:</strong> High-dimensional data</p>
                        <p><strong>Strengths:</strong> Excellent for text</p>
                        <p><strong>Use case:</strong> Maximum margin classification</p>
                    </div>
                    """, unsafe_allow_html=True)



            # Project features overview
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #ffecd2 0%, #fcb045 100%);
                padding: 25px;
                border-radius: 15px;
                margin: 30px 0;
                color: #333;
            ">
                <h3 style="text-align: center; margin-bottom: 20px;">🌟 Project Features</h3>
                <div style="display: flex; flex-wrap: wrap; justify-content: space-around;">
                    <div style="text-align: center; margin: 10px;">
                        <h4>🔍 Text Analysis</h4>
                        <p>Advanced NLP preprocessing</p>
                    </div>
                    <div style="text-align: center; margin: 10px;">
                        <h4>🤖 Multiple Models</h4>
                        <p>3 ML algorithms comparison</p>
                    </div>
                    <div style="text-align: center; margin: 10px;">
                        <h4>📊 Real-time Prediction</h4>
                        <p>Instant spam detection</p>
                    </div>
                    <div style="text-align: center; margin: 10px;">
                        <h4>🛡️ Security Tips</h4>
                        <p>Educational content</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Final Summary
            st.markdown("""
            ---
            <div style="
                background: linear-gradient(135deg, #000000 0%, #DE238A 100%);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin: 20px 0;
            ">
                <h3>🎉 Thank you for exploring our Spam Detection System!</h3>
                <p>This project demonstrates the power of Machine Learning in cybersecurity applications, 
                combining multiple algorithms with an intuitive web interface for real-world spam detection.</p>
                <p><strong>Built with ❤️ using Python, Streamlit, and Scikit-learn</strong></p>
            </div>
            """, unsafe_allow_html=True)

            #  # Academic note
            # st.info("📝 **Academic Note:** This project demonstrates practical application of machine learning concepts learned in Data Science curriculum, focusing on text classification and natural language processing techniques.")
 

if hasattr(st, 'cache_data'):
    try:
        # Check if we need to clear cache due to format changes
        if 'model_scores' in st.session_state:
            first_model_scores = next(iter(st.session_state.model_scores.values()), {})
            if 'f1_score' not in first_model_scores:
                # Clear old cache
                st.cache_data.clear()
                st.cache_resource.clear()
                # Clear session state
                for key in ['trained_models', 'model_scores', 'X_test', 'y_test']:
                    if key in st.session_state:
                        del st.session_state[key]
    except Exception:
        pass

# Alternative simple fix - add a button to manually clear cache in sidebar:
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Clear Cache & Retrain"):
    st.cache_data.clear()
    st.cache_resource.clear()
    for key in ['trained_models', 'model_scores', 'X_test', 'y_test']:
        if key in st.session_state:
            del st.session_state[key]
    st.sidebar.success("Cache cleared! Try prediction again.")
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    <p> Spam Detection System | Machine Learning</p>
    <p>Protecting your inbox with intelligent ML-powered detection</p>
</div>
""", unsafe_allow_html=True)