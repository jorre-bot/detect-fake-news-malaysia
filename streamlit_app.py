import streamlit as st
try:
    import streamlit_authenticator as stauth
except ImportError:
    st.error("Error: Could not import streamlit_authenticator. Please make sure it's installed correctly.")
    st.stop()

try:
    import yaml
    from yaml.loader import SafeLoader
    from predict_function import predict_news
    import os
    from datetime import datetime
    import pandas as pd
    import sqlite3
    import nltk
    import joblib
    
    # Download required NLTK data
    try:
        # Create nltk_data directory if it doesn't exist
        nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
        if not os.path.exists(nltk_data_dir):
            os.makedirs(nltk_data_dir)
        
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # Download required NLTK resources
        for resource in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        st.error(f"Error downloading NLTK data: {str(e)}")
        st.stop()
except ImportError as e:
    st.error(f"Error importing required packages: {str(e)}")
    st.stop()

# Initialize session state
if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=['Date', 'Text', 'Prediction', 'Confidence'])

# Database initialization
def init_db():
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS user_history
                     (username TEXT, date TEXT, text TEXT, prediction TEXT, confidence TEXT)''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")

def save_to_history(username, text, prediction, confidence):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO user_history VALUES (?, ?, ?, ?, ?)",
             (username, date, text, prediction, str(confidence)))
    conn.commit()
    conn.close()

def load_user_history(username):
    conn = sqlite3.connect('users.db')
    df = pd.read_sql_query("SELECT date as Date, text as Text, prediction as Prediction, confidence as Confidence FROM user_history WHERE username = ? ORDER BY date DESC", 
                          conn, params=(username,))
    conn.close()
    return df

# Initialize database
init_db()

# Set page config
st.set_page_config(
    page_title="Pengesan Berita Palsu Malaysia",
    page_icon="🔍",
    layout="centered"
)

# Load configuration file
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except Exception as e:
    st.error(f"Error loading config file: {str(e)}")
    st.stop()

# Create authenticator object
try:
    authenticator = stauth.Authenticate(
        credentials=config['credentials'],
        cookie_name=config['cookie']['name'],
        key=config['cookie']['key'],
        cookie_expiry_days=config['cookie']['expiry_days']
    )
except Exception as e:
    st.error(f"Authentication initialization error: {str(e)}")
    st.stop()

# Debug information
st.sidebar.write("Debug Information:")
st.sidebar.write(f"Current directory: {os.getcwd()}")
st.sidebar.write(f"Files in directory: {os.listdir()}")

try:
    # Check if model file exists
    model_path = 'best_fake_news_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
except Exception as e:
    st.error(f"Error checking model file: {str(e)}")

# Add custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-text {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .real {
        background-color: #d4edda;
        color: #155724;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Authentication
if not st.session_state['login_status']:
    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Log Masuk", "Daftar"])
    
    with tab1:
        try:
            name, authentication_status, username = authenticator.login("Log Masuk", location="main")
            if authentication_status:
                st.session_state['login_status'] = True
                st.session_state['username'] = username
                st.rerun()
            elif authentication_status == False:
                st.error('Username/password is incorrect')
            elif authentication_status == None:
                st.warning('Please enter your username and password')
        except Exception as e:
            st.error(f"Login error: {str(e)}")

    with tab2:
        try:
            if authenticator.register_user("Register user", preauthorization=False, location="main"):
                st.success('User registered successfully')
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(f"Registration error: {str(e)}")

else:
    # Show logout button in sidebar
    try:
        authenticator.logout("Logout", location="sidebar")
        st.sidebar.title(f'Selamat Datang, {st.session_state["username"]}!')
    except Exception as e:
        st.error(f"Logout error: {str(e)}")

    # Main app
    st.title("Pengesan Berita Palsu 🔍")

    # Tabs for Analysis and History
    tab1, tab2 = st.tabs(["Analisis", "Sejarah"])

    with tab1:
        # Text input
        news_text = st.text_area(
            "Masukkan Teks Berita:",
            height=200,
            placeholder="Taipkan atau tampal teks berita di sini..."
        )

        # Create two columns for buttons
        col1, col2 = st.columns(2)

        # Add buttons
        if col1.button("Analisis Berita", use_container_width=True):
            if news_text.strip():
                with st.spinner('Sedang menganalisis...'):
                    try:
                        # Add debug information
                        st.write("Debug Information:")
                        st.write("Text length:", len(news_text))
                        st.write("Sample text:", news_text[:100] + "..." if len(news_text) > 100 else news_text)
                        
                        # Make prediction
                        result = predict_news(news_text)
                        
                        if result['prediction'] != 'Error':
                            # Save to database
                            save_to_history(
                                st.session_state["username"],
                                news_text,
                                result['prediction'],
                                f"{result['confidence']:.2f}%"
                            )
                            
                            # Update session state history
                            st.session_state['history'] = load_user_history(st.session_state["username"])
                            
                            # Display results
                            st.markdown("### Keputusan Analisis:")
                            
                            # Create result container
                            result_class = "real" if result['prediction'] == "Real" else "fake"
                            st.markdown(f"""
                            <div class="prediction-text {result_class}">
                                Ramalan: {result['prediction']}<br>
                                Tahap Keyakinan: {result['confidence']:.2f}%
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("Maaf, terdapat masalah semasa menganalisis teks. Sila cuba lagi.")
                            
                    except Exception as e:
                        st.error(f"Error analyzing news: {str(e)}")
            else:
                st.warning("Sila masukkan teks berita untuk dianalisis.")

        if col2.button("Kosongkan", use_container_width=True):
            # This will trigger a rerun with empty text
            st.experimental_rerun()

    with tab2:
        # Load user history from database
        user_history = load_user_history(st.session_state["username"])
        if not user_history.empty:
            st.dataframe(user_history, use_container_width=True)
        else:
            st.info("Tiada sejarah analisis setakat ini.")

    # Footer
    st.markdown("---")
    st.markdown("© 2024 Pengesan Berita Palsu Malaysia. Semua hak cipta terpelihara.") 