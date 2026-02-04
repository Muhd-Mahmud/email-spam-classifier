#!/usr/bin/env python3
"""Test if everything is set up correctly"""

import sys
print("=" * 50)
print("Testing Python Development Environment")
print("=" * 50)

# Test 1: Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# Test 2: Import all required packages
print("\n📦 Testing package imports...")
packages_status = []

try:
    import pandas as pd
    packages_status.append(f"✓ pandas {pd.__version__}")
except ImportError:
    packages_status.append("✗ pandas - NOT FOUND")

try:
    import numpy as np
    packages_status.append(f"✓ numpy {np.__version__}")
except ImportError:
    packages_status.append("✗ numpy - NOT FOUND")

try:
    import matplotlib
    packages_status.append(f"✓ matplotlib {matplotlib.__version__}")
except ImportError:
    packages_status.append("✗ matplotlib - NOT FOUND")

try:
    import seaborn as sns
    packages_status.append(f"✓ seaborn {sns.__version__}")
except ImportError:
    packages_status.append("✗ seaborn - NOT FOUND")

try:
    import sklearn
    packages_status.append(f"✓ scikit-learn {sklearn.__version__}")
except ImportError:
    packages_status.append("✗ scikit-learn - NOT FOUND")

for status in packages_status:
    print(f"  {status}")

# Test 3: Quick ML test
print("\n🧪 Testing ML functionality...")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    
    # Simple test
    texts = ["hello world", "machine learning", "python programming"]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    print(f"  ✓ TF-IDF works! Shape: {X.shape}")
    
    model = MultinomialNB()
    model.fit(X, [0, 1, 0])
    print(f"  ✓ Model training works!")
    
except Exception as e:
    print(f"  ✗ ML test failed: {e}")

print("\n" + "=" * 50)
print(" Setup Complete! You're ready to code!")
print("=" * 50)