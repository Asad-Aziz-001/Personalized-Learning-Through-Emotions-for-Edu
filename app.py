from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import os
from datetime import datetime # <--- ADDED: For logging timestamps
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# --- Graphing Libraries ---
import shap
import matplotlib
matplotlib.use('Agg') # Prevent GUI errors
import matplotlib.pyplot as plt
import io
import base64
import traceback 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

# =========================================
#           DATABASE MODELS
# =========================================

# ---- User Model (UNCHANGED) ----
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    name = db.Column(db.String(150), nullable=False)
    saved_level = db.Column(db.String(50), nullable=True)
    completed_chapters = db.Column(db.String(50), default="")
    shap_plot = db.Column(db.Text, nullable=True)
    shap_text = db.Column(db.String(500), nullable=True)

# ---- [NEW] Behavior Log Model (ADDITION) ----
# This stores the rage clicks and mood data without altering the User table
class BehaviorLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    trigger_reason = db.Column(db.String(50))  # e.g., 'frustration', 'accuracy_fail'
    details = db.Column(db.String(200))        # e.g., 'Mood: stressed, Clicks: 6'

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ---- Load Model ----
# ---- Load Model (Extract if zipped) ----
import zipfile

ZIP_FILE = "depression_model_final.zip"   # Your zip file containing depression_model_final.joblib
MODEL_FILE = "depression_model_final.joblib"
model = None

# Extract model if ZIP exists
if os.path.exists(ZIP_FILE):
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall()  # extracts the .joblib file into the same directory
        print(f"SUCCESS: Extracted {ZIP_FILE}")
    except Exception as e:
        print(f"ERROR: Failed to extract ZIP. {e}")

# Load model after extraction
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        print(f"SUCCESS: {MODEL_FILE} loaded.")
    except Exception as e:
        print(f"ERROR: Failed to load model. {e}")
else:
    print("ERROR: Model file not found after extraction.")

# =========================================
#           AUTHENTICATION ROUTES
# =========================================

@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.json
    user = User.query.filter_by(username=data.get('username')).first()
    if user: return jsonify({"error": "User already exists"}), 400
    new_user = User(username=data.get('username'), name=data.get('name'), password=generate_password_hash(data.get('password'), method='pbkdf2:sha256'))
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "Account created!"})

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data.get('username')).first()
    if user and check_password_hash(user.password, data.get('password')):
        login_user(user)
        if user.saved_level: return jsonify({"redirect": "/dashboard", "status": "existing_user"})
        else: return jsonify({"redirect": "/survey", "status": "new_user"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# =========================================
#              PAGE ROUTES
# =========================================

@app.route('/')
def home(): return render_template('index.html') 
@app.route('/about')
def about(): return render_template('about.html')
@app.route('/features')
def features(): return render_template('SentellectPlatformFeatures.html')
@app.route('/login')
def login_page(): return render_template('login.html')
@app.route('/signup')
def signup_page(): return render_template('signup.html')
@app.route('/survey')
@login_required
def survey_page(): return render_template('questionnaire.html')

@app.route('/dashboard')
@login_required
def dashboard():
    level = current_user.saved_level if current_user.saved_level else 'mid'
    label = level.capitalize()
    shap_image = current_user.shap_plot
    shap_text = current_user.shap_text
    
    completed_list = current_user.completed_chapters.split(',') if current_user.completed_chapters else []
    completed_list = [x for x in completed_list if x]
    progress_percent = int((len(completed_list) / 6) * 100)

    return render_template('dashboard.html', 
                           level=level, 
                           label=label, 
                           progress=progress_percent, 
                           completed_chapters=completed_list, 
                           shap_image=shap_image,
                           shap_text=shap_text)

@app.route('/profile')
@login_required
def profile(): return render_template('profile.html', user=current_user)

# =========================================
#           PREDICTION & SHAP LOGIC
# =========================================

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model: return jsonify({"error": "Model not loaded"}), 500

    try:
        req_data = request.get_json()
        features = req_data.get("features")
        
        cols = ['Q3A','Q5A','Q10A','Q13A','Q16A','Q17A','Q21A','Q24A','Q26A','Q31A','Q34A','Q37A','Q38A','Q42A','Q2A','Q4A','Q7A','Q9A','Q15A','Q19A','Q20A','Q23A','Q25A','Q28A','Q30A','Q36A','Q40A','Q41A','age','gender']
        
        # Map IDs to readable names for the text explanation
        q_map = {'Q3A':"Positive Feeling", 'Q5A':"Initiative", 'Q10A':"Future Hope", 'Q13A':"Sadness", 'Q16A':"Interest", 'Q17A':"Self Worth", 'Q2A':"Dry Mouth", 'Q4A':"Breathing", 'Q7A':"Trembling", 'age':"Age", 'gender':"Gender"}

        df = pd.DataFrame([features], columns=cols)
        
        # 1. Predict
        prediction = model.predict(df)[0]
        pred_str = str(prediction).strip()
        print(f"Model Prediction: {pred_str}")

        # 2. Generate SHAP
        try:
            print("Attempting SHAP (Pipeline Mode)...")
            plt.clf()
            
            # --- PIPELINE SPLIT ---
            step_names = list(model.named_steps.keys())
            preprocessor = model.named_steps[step_names[0]]
            classifier = model.named_steps[step_names[-1]]
            
            # Transform input data
            X_trans = preprocessor.transform(df)
            if hasattr(X_trans, "toarray"): X_trans = X_trans.toarray()
            
            # Get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"Feature {i}" for i in range(X_trans.shape[1])]
            
            # Explain the CLASSIFIER part only
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(X_trans)
            shap_values.feature_names = list(feature_names)

            # --- HANDLE MULTICLASS ---
            if len(shap_values.shape) == 3:
                classes = classifier.classes_ 
                class_idx = list(classes).index(pred_str)
                vals = shap_values.values[0, :, class_idx]
                shap_values = shap_values[:, :, class_idx]
            else:
                vals = shap_values.values[0]

            # --- TEXT EXPLANATION ---
            max_idx = np.argmax(np.abs(vals))
            raw_feat = feature_names[max_idx]
            clean_feat = raw_feat.split('__')[-1] 
            readable_name = q_map.get(clean_feat, clean_feat)
            
            impact = "increased" if vals[max_idx] > 0 else "decreased"
            current_user.shap_text = f"The factor <b>'{readable_name}'</b> had the strongest influence and <b>{impact}</b> your result."

            # --- PLOT ---
            plt.figure(figsize=(10, 6), dpi=100)
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                 shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            else:
                 shap.plots.waterfall(shap_values, max_display=10, show=False)
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            current_user.shap_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            print("✅ SHAP SUCCESS")

        except Exception as e:
            print(f"⚠️ SHAP ERROR: {e}")
            traceback.print_exc()
            current_user.shap_plot = None
            current_user.shap_text = None

        # 3. Save Level
        if pred_str in ['High', 'Severe', 'Extremely Severe']: user_level = "high"
        elif pred_str in ['Mid', 'Moderate', 'Medium']: user_level = "mid"
        else: user_level = "low"

        current_user.saved_level = user_level
        db.session.commit()

        return jsonify({"prediction": pred_str, "redirect_url": "/dashboard"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================================
#           QUIZ & CHAPTER ROUTES
# =========================================

@app.route('/quiz/<int:chapter_num>')
@login_required
def serve_quiz(chapter_num): return render_template(f'Quiz{chapter_num}.html', chapter=chapter_num)

@app.route('/api/complete_chapter', methods=['POST'])
@login_required
def complete_chapter():
    data = request.json
    chapter = str(data.get('chapter'))
    current = current_user.completed_chapters.split(',') if current_user.completed_chapters else []
    if chapter not in current:
        current.append(chapter)
        current_user.completed_chapters = ",".join([x for x in current if x])
        db.session.commit()
    return jsonify({"success": True, "redirect": "/dashboard"})

# [UPDATED] Route to handle Adaptive Template Switching
# REPLACE THIS FUNCTION IN app.py
@app.route('/chapter/<int:chapter_num>')
@login_required
def serve_chapter(chapter_num):
    # 1. Debug Input
    requested_mode = request.args.get('mode')
    print(f"🔍 DEBUG: Link clicked with mode = {requested_mode}")

    if requested_mode in ['low', 'mid', 'high']:
        current_user.saved_level = requested_mode
        db.session.commit()
        print(f"✅ DEBUG: Database updated to level: {current_user.saved_level}")

    # 2. Debug Level Loading
    level = current_user.saved_level if current_user.saved_level else 'mid'
    target_file = f"chapter{chapter_num}_{level}.html"
    print(f"📂 DEBUG: Trying to open file: {target_file}")
    
    # 3. Serve with Error Printing
    try:
        return render_template(target_file)
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not find or load {target_file}")
        print(f"❌ ERROR DETAILS: {e}")
        print("⚠️ Falling back to MID template")
        return render_template(f"chapter{chapter_num}_mid.html")

@app.route('/chapter/<int:chapter_num>/part/<int:part_num>')
@login_required
def serve_chapter_part(chapter_num, part_num):
    level = current_user.saved_level if current_user.saved_level else 'mid'
    return render_template(f"chapter{chapter_num}_{level}{part_num}.html")

# [NEW] Route to Log Adaptive Behavior (Frontend -> Database)
@app.route('/api/log_behavior', methods=['POST'])
@login_required
def log_behavior():
    try:
        data = request.json
        reason = data.get('reason', 'unknown')
        mood = data.get('mood', 'neutral')
        rage_clicks = data.get('rageClicks', 0)
        
        # Create a new log entry
        new_log = BehaviorLog(
            user_id=current_user.id,
            trigger_reason=reason,
            details=f"Mood: {mood}, RageClicks: {rage_clicks}"
        )
        
        db.session.add(new_log)
        db.session.commit()
        
        return jsonify({"status": "success"})
    
    except Exception as e:
        print(f"Log Error: {e}")
        return jsonify({"status": "error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)



