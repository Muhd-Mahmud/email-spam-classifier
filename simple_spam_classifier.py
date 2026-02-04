"""
TASK 1:
Email Spam Classifier
Author: [Muhammed Mahmud Yahman]
Date: F4 ebruary 2026

ARCH TECHNOLOGIES INTERSHIP
AI/ML ENGINEERING INTERN

A machine learning project to classify emails as spam or ham.
This uses the SMS Spam Collection dataset for training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import re
import string
import warnings
warnings.filterwarnings('ignore')

print("Email Spam Classification Project")
print("-" * 50)

# Load the dataset
print("\nLoading dataset...")
try:
    # Try to load from local file first
    df = pd.read_csv('spam.csv', encoding='latin-1')
    # The dataset has some extra columns, we only need the first two
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    print(f"Loaded {len(df)} messages from spam.csv")
except FileNotFoundError:
    # If local file doesn't exist, download it
    print("Local file not found. Downloading dataset...")
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    # Save it locally for next time
    df.to_csv('spam_data.csv', index=False)
    print(f"Downloaded and saved {len(df)} messages")

# Quick look at the data
print("\nFirst few examples:")
print(df.head(3))

print(f"\nDataset info:")
print(f"Total messages: {len(df)}")
print(f"Spam: {sum(df['label'] == 'spam')} ({sum(df['label'] == 'spam')/len(df)*100:.1f}%)")
print(f"Ham: {sum(df['label'] == 'ham')} ({sum(df['label'] == 'ham')/len(df)*100:.1f}%)")

# Text preprocessing
def clean_text(text):
    """Clean up the email text"""
    # lowercase everything
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove extra spaces
    text = ' '.join(text.split())
    return text

print("\nCleaning text data...")
df['cleaned_text'] = df['message'].apply(clean_text)

# Example of before/after cleaning
print("\nCleaning example:")
sample_idx = 10
print(f"Original: {df['message'].iloc[sample_idx]}")
print(f"Cleaned: {df['cleaned_text'].iloc[sample_idx]}")

# Prepare data for training
X = df['cleaned_text']
y = df['label'].map({'ham': 0, 'spam': 1})

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nSplit data into:")
print(f"Training: {len(X_train)} messages")
print(f"Testing: {len(X_test)} messages")

# Convert text to numerical features using TF-IDF
print("\nConverting text to features...")
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), 
                             min_df=2, max_df=0.8, stop_words='english')

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

print(f"Created {X_train_features.shape[1]} features")

# Train multiple models
print("\nTraining models...")
print("-" * 50)

models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    model.fit(X_train_features, y_train)
    predictions = model.predict(X_test_features)
    
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    results[name] = {
        'model': model,
        'predictions': predictions,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
    
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall: {rec:.3f}")
    print(f"  F1-Score: {f1:.3f}")

# Compare models
print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()]
})

print(comparison.to_string(index=False))

# Find best model
best_model_name = comparison.loc[comparison['Accuracy'].idxmax(), 'Model']
print(f"\nBest performing model: {best_model_name}")
print("=" * 50)

# Detailed evaluation of best model
best_model = results[best_model_name]['model']
best_preds = results[best_model_name]['predictions']

print(f"\nDetailed results for {best_model_name}:")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_preds)
print(f"              Predicted")
print(f"           Ham    Spam")
print(f"Ham      {cm[0][0]:5d}  {cm[0][1]:5d}")
print(f"Spam     {cm[1][0]:5d}  {cm[1][1]:5d}")

print("\nClassification Report:")
print(classification_report(y_test, best_preds, 
                          target_names=['Ham', 'Spam']))

# Test with custom examples
print("\n" + "=" * 50)
print("TESTING WITH CUSTOM MESSAGES")
print("=" * 50)

test_messages = [
    "Hey, are you free for lunch tomorrow?",
    "WINNER! You have won a $1000 Walmart gift card. Click here NOW!",
    "Can you send me the meeting notes from yesterday?",
    "URGENT: Your account has been compromised. Verify your identity immediately!",
    "Thanks for helping me with the project, really appreciate it",
    "Congratulations! You've been selected for a FREE cruise to the Bahamas!"
]

print("\nTesting custom messages:")
for i, msg in enumerate(test_messages, 1):
    cleaned = clean_text(msg)
    features = vectorizer.transform([cleaned])
    prediction = best_model.predict(features)[0]
    label = "SPAM" if prediction == 1 else "HAM"
    
    # Get confidence if available
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(features)[0]
        confidence = proba[prediction] * 100
        print(f"\n{i}. {msg[:60]}...")
        print(f"   Prediction: {label} (confidence: {confidence:.1f}%)")
    else:
        print(f"\n{i}. {msg[:60]}...")
        print(f"   Prediction: {label}")

# Create visualizations
print("\n" + "=" * 50)
print("Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Spam Classification Analysis', fontsize=16, fontweight='bold')

# 1. Model comparison
ax1 = axes[0, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x_pos = np.arange(len(comparison))
width = 0.2

for i, metric in enumerate(metrics):
    ax1.bar(x_pos + i*width, comparison[metric], width, label=metric)

ax1.set_xlabel('Models')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison')
ax1.set_xticks(x_pos + width * 1.5)
ax1.set_xticklabels(comparison['Model'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Confusion matrix heatmap
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
ax2.set_title(f'Confusion Matrix - {best_model_name}')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

# 3. Class distribution
ax3 = axes[1, 0]
labels = ['Ham', 'Spam']
sizes = [sum(y == 0), sum(y == 1)]
colors = ['lightgreen', 'lightcoral']
ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
ax3.set_title('Dataset Distribution')

# 4. Message length distribution
ax4 = axes[1, 1]
df['msg_length'] = df['message'].str.len()
ham_lengths = df[df['label'] == 'ham']['msg_length']
spam_lengths = df[df['label'] == 'spam']['msg_length']

ax4.hist(ham_lengths, bins=30, alpha=0.6, label='Ham', color='green')
ax4.hist(spam_lengths, bins=30, alpha=0.6, label='Spam', color='red')
ax4.set_xlabel('Message Length')
ax4.set_ylabel('Frequency')
ax4.set_title('Message Length Distribution')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('spam_analysis.png', dpi=300, bbox_inches='tight')
print("Saved visualization to spam_analysis.png")

# Summary
print("\n" + "=" * 50)
print("PROJECT SUMMARY")
print("=" * 50)
print(f"""
Dataset: {len(df)} messages ({sum(y==1)} spam, {sum(y==0)} ham)
Features extracted: {X_train_features.shape[1]} TF-IDF features
Models trained: {len(models)}
Best model: {best_model_name}
Test accuracy: {results[best_model_name]['accuracy']:.1%}

Key findings:
- {best_model_name} performed best with {results[best_model_name]['accuracy']:.1%} accuracy
- Precision: {results[best_model_name]['precision']:.1%} (low false positives)
- Recall: {results[best_model_name]['recall']:.1%} (catches most spam)
""")
print("=" * 50)

print("\nDone! Check spam_analysis.png for visualizations.")