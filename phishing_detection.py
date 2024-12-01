import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
from flask import Flask, request, render_template
from urllib.parse import urlparse
import re
from scipy.sparse import hstack

kaggle_data = pd.read_csv(r'D:\CNS Projects\CNS 1\dataset.csv')
phishing_data = pd.read_csv(r'D:\CNS Projects\CNS 1\phishing_data.csv')

vectorizer = CountVectorizer()

def get_scheme(url):
    if isinstance(url, str):
        parsed_url = urlparse(url)
        return 1 if parsed_url.scheme == 'https' else 0
    return 0

def has_ip_address(url):
    return 1 if isinstance(url, str) and re.match(r"http[s]?://\d+\.\d+\.\d+\.\d+", url) else 0

def has_suspicious_subdomain(url):
    if isinstance(url, str):
        parsed_url = urlparse(url)
        return 1 if len(parsed_url.netloc.split('.')) > 3 else 0
    return 0

def has_keywords(url):
    return 1 if isinstance(url, str) and re.search(r'(login|verify|bank|secure|account)', url, re.IGNORECASE) else 0

def is_shortened_url(url):
    return 1 if isinstance(url, str) and re.match(r"http[s]?://(bit\.ly|tinyurl\.com|goo\.gl|t\.co)", url) else 0

kaggle_data['URL'] = kaggle_data['URL'].fillna('')
kaggle_data['https'] = kaggle_data['URL'].apply(get_scheme)
kaggle_data['ip_address'] = kaggle_data['URL'].apply(has_ip_address)
kaggle_data['suspicious_subdomain'] = kaggle_data['URL'].apply(has_suspicious_subdomain)
kaggle_data['keywords_in_url'] = kaggle_data['URL'].apply(has_keywords)
kaggle_data['shortened_url'] = kaggle_data['URL'].apply(is_shortened_url)

phishing_data['url'] = phishing_data['url'].fillna('')
phishing_data['https'] = phishing_data['url'].apply(get_scheme)
phishing_data['ip_address'] = phishing_data['url'].apply(has_ip_address)
phishing_data['suspicious_subdomain'] = phishing_data['url'].apply(has_suspicious_subdomain)
phishing_data['keywords_in_url'] = phishing_data['url'].apply(has_keywords)
phishing_data['shortened_url'] = phishing_data['url'].apply(is_shortened_url)

combined_data = pd.concat([kaggle_data[['URL', 'https', 'ip_address', 'suspicious_subdomain', 'keywords_in_url', 'shortened_url', 'Type']],
                           phishing_data[['url', 'https', 'ip_address', 'suspicious_subdomain', 'keywords_in_url', 'shortened_url', 'label']].rename(columns={'url': 'URL', 'label': 'Type'})],
                          ignore_index=True)

X_text_features = vectorizer.fit_transform(combined_data['URL'])
X_other_features = np.array(combined_data[['https', 'ip_address', 'suspicious_subdomain', 'keywords_in_url', 'shortened_url']])
X = hstack([X_text_features, X_other_features])

y = combined_data['Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    url_text_features = vectorizer.transform([url])
    url_other_features = np.array([[get_scheme(url), has_ip_address(url), has_suspicious_subdomain(url), has_keywords(url), is_shortened_url(url)]])
    url_features = hstack([url_text_features, url_other_features])

    prediction = model.predict(url_features)
    result = 'Phishing URL' if prediction[0] == 1 else 'Legitimate URL'
    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
