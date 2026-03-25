from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import os
import random
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ==========================================
# DYNAMIC PATH CONFIGURATION
# ==========================================
# This sets the paths dynamically according to VS Code current working directory, so it will work for everyone without needing to change any hardcoded paths.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 1. MongoDB Connection Setup
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    db = client['ecommerce_db']
    collection = db['user_predictions']
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {e}")

# 2. Load the Model and Encoders
try:
    model = joblib.load(os.path.join(SRC_DIR, 'random_forest_model_tuned.pkl'))
    scaler = joblib.load(os.path.join(SRC_DIR, 'standard_scaler.pkl'))
    encoders = joblib.load(os.path.join(SRC_DIR, 'label_encoders.pkl'))
    print("✅ ML Models and Encoders loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "E-commerce API is running! 🚀"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # A. Get data from website
        data = request.json
        df = pd.DataFrame([data])

        # B. Encode and Scale
        categorical_cols = ['gender', 'product_category', 'brand', 'customer_segment', 'payment_method', 'city', 'season']
        for col in categorical_cols:
            if data[col] in encoders[col].classes_:
                df[col] = encoders[col].transform([data[col]])[0]
            else:
                df[col] = encoders[col].transform([encoders[col].classes_[0]])[0]

        numerical_cols = ['age', 'price', 'discount', 'rating', 'days_since_last_purchase', 'units_sold', 'delivery_days', 'review_count']
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # C. Make Prediction
        prediction = model.predict(df)
        probability = model.predict_proba(df)[0][1]
        result_text = "Will Purchase" if prediction[0] == 1 else "Will Not Purchase"
        probability_pct = round(float(probability) * 100, 2)

        # D. SAVE TO MONGODB
        try:
            db_record = data.copy()
            db_record['ai_prediction_result'] = result_text
            db_record['ai_probability'] = probability_pct
            db_record['timestamp'] = datetime.now()
            collection.insert_one(db_record)
            print("💾 New record saved to MongoDB!")
        except Exception as mongo_err:
            print(f"⚠️ MongoDB Save Error: {mongo_err}")

        # E. AUTO-UPDATE CSV FOR POWER BI
        try:
            # 1. Force create the 'data' directory if it doesn't exist
            os.makedirs(DATA_DIR, exist_ok=True) 
            
            csv_file_path = os.path.join(DATA_DIR, 'cleaned_clothing_dataset.csv')
            
            new_csv_row = {
                'order_id': random.randint(200001, 999999),
                'customer_id': random.randint(50001, 99999),
                'age': data['age'],
                'gender': data['gender'],
                'product_category': data['product_category'],
                'brand': data['brand'],
                'price': data['price'],
                'discount': data['discount'],
                'rating': data['rating'],
                'days_since_last_purchase': data['days_since_last_purchase'],
                'units_sold': data['units_sold'],
                'customer_segment': data['customer_segment'],
                'payment_method': data['payment_method'],
                'city': data['city'],
                'season': data['season'],
                'return_customer': data['return_customer'],
                'delivery_days': data['delivery_days'],
                'stock_available': data['stock_available'],
                'review_count': data['review_count'],
                'will_purchase': int(prediction[0]), 
                'order_date': datetime.now().strftime('%d-%m-%Y %H:%M')
            }
            
            df_new = pd.DataFrame([new_csv_row])
            
            # 2. Check if file exists. If not, write headers too.
            file_exists = os.path.isfile(csv_file_path)
            df_new.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)
            
            print(f"📊 Success! Data appended to: {csv_file_path}")
            
        except Exception as csv_error:
            print(f"⚠️ Could not update CSV: {csv_error}")

        # F. Return answer to website
        return jsonify({
            'prediction': int(prediction[0]),
            'status': result_text,
            'purchase_probability': probability_pct
        })

    except Exception as e:
        print(f"❌ Critical Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
    