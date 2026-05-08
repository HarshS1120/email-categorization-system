# main.py - Complete Email Categorization System
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from datetime import datetime

# Add this at the top of main.py, right after imports
import os

def ensure_dataset_exists():
    """Create dataset if it doesn't exist"""
    if not os.path.exists('enron_dataset.csv'):
        print("Dataset not found. Creating dataset...")
        import pandas as pd
        import random
        
        # Email templates for different categories
        work_emails = [
            "Project deadline extended to Friday. Please update your tasks accordingly.",
            "Team meeting at 3 PM in Conference Room A to discuss Q4 goals.",
            "Can you review the attached proposal and share feedback by EOD?",
            "Budget approval received from finance for the new initiative.",
            "Server maintenance scheduled for Saturday 2 AM - 4 AM.",
            "Quarterly report needs to be submitted by Monday morning.",
        ]
        
        personal_emails = [
            "Hey! Want to grab coffee later today? Haven't seen you in ages!",
            "Don't forget about dinner at my place tomorrow at 7 PM.",
            "Happy Birthday! Hope you have an amazing day! 🎂",
            "Movie night on Friday? New superhero film is out!",
            "Working from home tomorrow, let's catch up over video call.",
            "Gym session at 6 PM? Need a workout buddy!",
        ]
        
        spam_emails = [
            "CONGRATULATIONS! You've won $1,000,000. Click here to claim now!",
            "URGENT: Your bank account has been compromised. Verify now at http://fake-link.com",
            "FREE VIAGRA and Cialis! 80% off today only!",
            "You have inherited $10,000,000 from a distant relative. Send bank details.",
            "Get rich quick! Work from home and earn $5000/week!",
            "Your Netflix account is suspended. Update payment here: http://scam-site.net",
        ]
        
        finance_emails = [
            "Your monthly statement is ready. Balance: $2,450.32",
            "Credit card payment due on 15th of this month.",
            "Invoice #INV-2024-001 attached for your records.",
            "Tax filing reminder: Deadline approaching in 30 days.",
            "Investment portfolio update for Q4 is now available.",
        ]
        
        emails = []
        categories = []
        
        for i in range(1000):
            if i < 400:
                category = "Work"
                text = random.choice(work_emails)
            elif i < 650:
                category = "Personal"
                text = random.choice(personal_emails)
            elif i < 850:
                category = "Spam"
                text = random.choice(spam_emails)
            else:
                category = "Finance"
                text = random.choice(finance_emails)
            
            emails.append(text)
            categories.append(category)
        
        df = pd.DataFrame({'email': emails, 'category': categories})
        df.to_csv('enron_dataset.csv', index=False)
        print(f"✅ Created dataset with {len(df)} emails")

# Call this function right after imports, before loading the dataset
ensure_dataset_exists()

# Download NLTK data (first time only)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

print("="*70)
print("📧 INTELLIGENT EMAIL CATEGORIZATION SYSTEM")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# 1. Load dataset
print("📂 STEP 1: Loading Dataset")
print("-"*40)
df = pd.read_csv('enron_dataset.csv')
print(f"✅ Loaded {len(df)} emails")
print(f"📊 Categories found: {df['category'].unique().tolist()}")
print(f"\nCategory distribution:")
for cat, count in df['category'].value_counts().items():
    print(f"   {cat}: {count} emails ({count/len(df)*100:.1f}%)")

# 2. Text preprocessing
print("\n🔧 STEP 2: Text Preprocessing")
print("-"*40)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and normalize email text"""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words and len(word) > 2])
    return text

# Apply preprocessing
df['cleaned_email'] = df['email'].apply(clean_text)

# Show example
print(f"Original: {df['email'].iloc[0][:80]}...")
print(f"Cleaned:  {df['cleaned_email'].iloc[0][:80]}...")

# 3. Feature extraction
print("\n📊 STEP 3: Feature Extraction (TF-IDF)")
print("-"*40)
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    stop_words='english'
)
X = vectorizer.fit_transform(df['cleaned_email'])
y = df['category']
print(f"✅ Created {X.shape[1]} features from {X.shape[0]} emails")

# 4. Train-test split
print("\n✂️ STEP 4: Train-Test Split")
print("-"*40)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape[0]} emails")
print(f"Test set: {X_test.shape[0]} emails")

# 5. Train models
print("\n🤖 STEP 5: Training Models")
print("-"*40)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Linear SVM': LinearSVC(max_iter=2000, random_state=42)
}

results = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred
    }
    print(f"   ✅ Accuracy: {accuracy:.2%}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

print(f"\n🏆 BEST MODEL: {best_name} with {best_accuracy:.2%} accuracy")

# 6. Detailed evaluation
print("\n📈 STEP 6: Model Evaluation")
print("-"*40)
y_pred = results[best_name]['predictions']
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save model
print("\n💾 STEP 7: Saving Model")
print("-"*40)
os.makedirs('models', exist_ok=True)

with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✅ Model saved to 'models/' directory")

# 8. Test on new emails
print("\n🔮 STEP 8: Testing on New Emails")
print("-"*40)

# Sample emails to test
test_emails = [
    "Meeting scheduled for tomorrow at 2pm to discuss the project roadmap",
    "Hey! Want to grab coffee this weekend? Haven't seen you in a while!",
    "CONGRATULATIONS! You have won $1,000,000! Click here to claim your prize",
    "Your monthly investment portfolio has increased by 12% this quarter",
    "Please review the attached contract and get back to me by Friday",
    "FREE OFFER: Get Viagra and Cialis at 80% discount! Limited time only",
    "Don't forget about the team dinner tomorrow at 7pm",
    "Your bank account has been compromised. Verify your details immediately"
]

# Load model for testing
with open('models/best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open('models/vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

print("\nPredictions for new emails:")
print("-"*60)

for email in test_emails:
    cleaned = clean_text(email)
    features = loaded_vectorizer.transform([cleaned])
    prediction = loaded_model.predict(features)[0]
    
    # Get confidence score if available
    if hasattr(loaded_model, 'predict_proba'):
        proba = loaded_model.predict_proba(features)[0]
        confidence = max(proba) * 100
        print(f"\n📧 {email[:50]}...")
        print(f"   → {prediction} (Confidence: {confidence:.1f}%)")
    else:
        print(f"\n📧 {email[:50]}... → {prediction}")

# 9. Save results summary
print("\n📝 STEP 9: Generating Report")
print("-"*40)
report = f"""
====================================
EMAIL CATEGORIZATION SYSTEM REPORT
====================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET INFO:
- Total emails: {len(df)}
- Categories: {', '.join(df['category'].unique())}
- Distribution: {dict(df['category'].value_counts())}

MODEL PERFORMANCE:
- Best Model: {best_name}
- Accuracy: {best_accuracy:.2%}
- Features used: {X.shape[1]}

MODEL FILES SAVED:
- models/best_model.pkl
- models/vectorizer.pkl

====================================
"""

with open('project_report.txt', 'w') as f:
    f.write(report)

print("✅ Report saved to 'project_report.txt'")

print("\n" + "="*70)
print("✅ SYSTEM READY! Email categorization successful!")
print("="*70)
print("\n📁 Files created:")
print("   - enron_dataset.csv (your dataset)")
print("   - models/best_model.pkl (trained model)")
print("   - models/vectorizer.pkl (feature extractor)")
print("   - project_report.txt (performance summary)")
print("\n🎯 Next steps:")
print("   1. Run the system anytime with: python main.py")
print("   2. We'll add a web interface next")
print("   3. We'll enable batch processing of multiple emails")