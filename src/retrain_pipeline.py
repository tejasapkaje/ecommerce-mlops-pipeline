import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Automated Retraining Pipeline...")

# ==========================================
# 1. PATH CONFIGURATION (Foolproof)
# ==========================================
# The file is in  'src' folder, so go back 1 step to set the path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cleaned_clothing_dataset.csv')

# ==========================================
# 2. LOAD THE FULL UPDATED DATA (Old + New)
# ==========================================
try:
    print(f"📂 Loading updated dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded successfully! Total records: {len(df)}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# ==========================================
# 3. PREPROCESSING (Feature Engineering)
# ==========================================
print("⚙️ Processing features...")
# Drop IDs & Dates as they don't help in prediction
columns_to_drop = ['order_id', 'customer_id', 'order_date']
df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)

# Encode Categorical Data
categorical_cols = ['gender', 'product_category', 'brand', 'customer_segment', 'payment_method', 'city', 'season']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Scale Numerical Data
numerical_cols = ['age', 'price', 'discount', 'rating', 'days_since_last_purchase', 'units_sold', 'delivery_days', 'review_count']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# ==========================================
# 4. TRAIN THE MODEL (The "No Forgetting" Step)
# ==========================================
print("🧠 Retraining the AI Model on full historical + new data...")
X = df.drop('will_purchase', axis=1)
y = df['will_purchase']

# Initialize the best model settings found in Hyperparameter tuning
rf_model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# ==========================================
# 5. SAVE NEW BRAIN & RULES
# ==========================================
print("💾 Saving the updated Model and Encoders...")
joblib.dump(rf_model, os.path.join(SRC_DIR, 'random_forest_model_tuned.pkl'))
joblib.dump(scaler, os.path.join(SRC_DIR, 'standard_scaler.pkl'))
joblib.dump(encoders, os.path.join(SRC_DIR, 'label_encoders.pkl'))

print("🎉 Retraining Complete! The API will now use the smarter model.")
