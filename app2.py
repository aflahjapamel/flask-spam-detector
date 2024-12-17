from flask import Flask, request, redirect, render_template, session, g
import os
import json
import joblib
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import string
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import emoji
import pandas as pd
import os
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'fallback_secret_key')
# Secure the session with a random secret key
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True




CLIENT_SECRETS_FILE = "credentials.json"  # File JSON kredensial yang diunduh dari GCP
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
REDIRECT_URI = "https://alproweb.uc.r.appspot.com/callback"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Load machine learning model and vectorizer
model = joblib.load(os.path.join(BASE_DIR, 'Logistic Regression.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'Logistic Regression_vectorizer.pkl'))

# Load slang dictionary and abbreviation mapping
kamus_alay = pd.read_csv(os.path.join(BASE_DIR, 'kamus_alay.csv'))  # Pastikan file tersedia
kamus_alay = kamus_alay.rename(columns={'slang': 'kamus_slang', 'formal': 'kamus_perbaikan'})
slang_mapping = dict(zip(kamus_alay['kamus_slang'], kamus_alay['kamus_perbaikan']))
kamus_singkatan = pd.read_csv(os.path.join(BASE_DIR,'kamus_singkatan.csv', header=None, names=['sebelum_perbaikan', 'setelah_perbaikan'], delimiter=';'))
singkatan_mapping = dict(zip(kamus_singkatan['sebelum_perbaikan'], kamus_singkatan['setelah_perbaikan']))

# Stopword remover
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
stopword_factory = StopWordRemoverFactory()
stopwords = stopword_factory.get_stop_words()

# Exclude important stopwords
excluded_stopwords = {"tidak", "bukan", "jangan", "tak", "belum"}
stopwords = set(stopwords).difference(excluded_stopwords)

# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Preprocessing function
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r'http[s]?://\S+', '', message)
    message = emoji.demojize(message)
    message = re.sub(r'[0-9]+', '', message)
    message = ' '.join([slang_mapping.get(word, word) for word in message.split()])
    message = ' '.join([singkatan_mapping.get(word, word) for word in message.split()])
    message = ''.join([char for char in message if char not in string.punctuation])
    message = ' '.join([word for word in message.split() if word not in stopwords])
    message = stemmer.stem(message)
    return message

# Predict spam
def predict_spam(message):
    preprocessed_message = preprocess_message(message)
    vectorized_message = vectorizer.transform([preprocessed_message])
    prediction = model.predict(vectorized_message)
    return "Spam" if prediction[0] == 1 else "Not Spam"

@app.route('/')
def index():
    if 'credentials' not in session:
        # Start the Gmail authorization process
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        auth_url, _ = flow.authorization_url(prompt='consent')
        return redirect(auth_url)  # Redirect to Google's authorization page
    else:
        # If user is already authenticated, show the HTML form
        return render_template('index.html')

# OAuth2 callback route to handle the redirect from Google
@app.route('/callback')
def callback():
    if request.args.get('code'):
        # Fetch the token using the authorization code
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        flow.fetch_token(authorization_response=request.url)
        
        # Save the credentials in the session
        credentials = flow.credentials
        session['credentials'] = credentials_to_dict(credentials)

        # Redirect to the index route to display the form
        return redirect('/')

    else:
        return "Missing code parameter in response.", 400

# Function to convert credentials object to dictionary
def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

# Function to load credentials from session or token file
def load_credentials():
    if 'credentials' in session:
        creds_data = session['credentials']
        creds = Credentials(
            creds_data['token'],
            refresh_token=creds_data['refresh_token'],
            token_uri=creds_data['token_uri'],
            client_id=creds_data['client_id'],
            client_secret=creds_data['client_secret'],
            scopes=creds_data['scopes']
        )
        return creds
    elif os.path.exists('token.json'):
        with open('token.json', 'r') as token_file:
            creds_data = json.load(token_file)
            creds = Credentials(
                creds_data['token'],
                refresh_token=creds_data['refresh_token'],
                token_uri=creds_data['token_uri'],
                client_id=creds_data['client_id'],
                client_secret=creds_data['client_secret'],
                scopes=creds_data['scopes']
            )
            return creds
    else:
        raise Exception("Token file not found. Please authenticate first.")

# Endpoint to handle fetching the number of emails
@app.route('/start', methods=['POST'])
def start():
    try:
        # Retrieve email count from form
        email_count = request.form.get('emailCount')
        session['email_count'] = email_count  # Save to session

        creds = load_credentials()  # Load credentials from session
        service = build('gmail', 'v1', credentials=creds)

        # Fetch the emails from Gmail
        results = service.users().messages().list(userId='me', maxResults=int(email_count)).execute()
        messages = results.get('messages', [])
        email_data = []
        for msg in messages:
            msg_detail = service.users().messages().get(userId='me', id=msg['id']).execute()
            snippet = msg_detail.get('snippet', 'No snippet available')
            
            spam_status = predict_spam(snippet)

            email_data.append({
                'snippet': snippet,
                'spam_status': spam_status,
                'class': 'spam' if spam_status == 'Spam' else 'not-spam'
            })

        return render_template('output.html', email_data=email_data)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run()