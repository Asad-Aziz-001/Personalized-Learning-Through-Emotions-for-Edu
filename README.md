
# 📘 **Sentellect — Personalized Learning for Education**

Sentellect is a **personalized learning platform** designed for **HSSC Part 2 students**.
It uses **machine learning (joblib model)** and a **Flask + HTML** framework to provide
customized learning recommendations and user-specific insights.

The project includes a backend model, a structured Flask environment,
and a database instance containing user records.

# **Demo Link**

# [![🚀 Live Demo – Railway](https://img.shields.io/badge/🚀-Live%20Demo%20%E2%80%93%20Railway-0D87E0?logo=railway&logoColor=white)](
https://personalized-learning-through-emotions-for-edu-production.up.railway.app/)


---

## 🚀 **Features**

* 🎓 Personalized study recommendations
* 🤖 Machine Learning model integrated using `.joblib`
* 🌐 Flask backend with clean HTML templates
* 👤 User database stored inside the `instances/` folder
* 📊 Dynamic content based on user inputs
* 🖥 Simple and lightweight interface for student usage

---

## 🧱 **Project Structure**

```
sentellect/
├── app.py
├── templates/
├── model.joblib
├── requirements.txt
├── instances/
│   ├── database.sqlite3 (example)
│   └── user_records/
```

---

## 🔧 **Installation Instructions**

Follow these steps to set up and run the project.

---

### **1️⃣ Clone the Repository**

```
git clone https://github.com/your-username/sentellect.git
cd sentellect
```

---

### **2️⃣ Create Virtual Environment**

```
python -m venv venv
```

Activate it:

#### **Windows**

```
venv\Scripts\activate
```

#### **PowerShell**

```
venv\Scripts\Activate.ps1
```

---

### **3️⃣ Install Dependencies**

Install all required packages:

```
pip install -r requirements.txt
```

---

### **4️⃣ Run Django Server**

Since your project includes `app.py`, run:

```
python app.py
```



## 🤖 **Machine Learning Model**

The project uses a `.joblib` ML model located at:

```
model.joblib
```

 Google Drive link here:

**Model Download:**
https://drive.google.com/file/d/1JqUMN4wUUzcHNXjbS4sl4cakNZ_dy6DP/view?usp=sharing

---

## 🗄️ **Database (instances folder)**

The `instances/` directory contains your user database and other stored files.

This allows:

* saving user details
* tracking progress
* storing personalized recommendations

Make sure **NOT** to delete this folder.

---

## 📄 **License**

This project is for educational and academic use.

---

## 🤝 **Contributing**

Pull requests and suggestions are welcome.


Just tell me!

