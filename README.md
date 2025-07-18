# **Student Performance Prediction – End-to-End ML Project**

## 📌 **Project Overview**
This project aims to predict students' exam scores based on various demographic and academic-related factors such as gender, parental education, lunch type, and test preparation. The solution is designed as an **end-to-end machine learning pipeline**, including **data ingestion, data transformation, model training, and deployment** using a Flask web application.

---

## ✅ **Problem Statement**
Educational institutions need a way to identify students who are likely to underperform so that timely interventions can be provided. Using past performance and related attributes, this project builds a predictive model that forecasts student scores.

---

## ✅ **Tech Stack**
- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Scikit-learn, CatBoost, XGBoost, Matplotlib, Seaborn
- **Model Deployment:** Flask
- **Containerization:** Docker
- **Version Control:** Git/GitHub

---

## ✅ **Project Workflow**
### **1. Data Ingestion**
- Load dataset from CSV
- Split into training and testing sets
- Save raw, train, and test data in `artifacts/`

### **2. Data Transformation**
- Handle missing values (if any)
- Encode categorical features using OneHotEncoder
- Scale numerical features using StandardScaler
- Create and save preprocessing pipeline

### **3. Model Training**
- Train multiple regression models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - CatBoost
  - AdaBoost
- Evaluate models using R² score and RMSE
- Select the best-performing model (CatBoost achieved highest R² ~0.88)
- Save the model as `model.pkl`

### **4. Deployment**
- Build a Flask-based web app
- Create HTML templates for user input and result display
- Integrate preprocessing pipeline and model for predictions

---

## ✅ **Project Architecture**
mlproject/
│
├── artifacts/ # Stores trained models & preprocessor
├── notebook/ # EDA & experiments
├── src/ # Core ML pipeline
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ ├── pipeline/
│ │ ├── train_pipeline.py
│ │ └── predict_pipeline.py
│ ├── utils.py
│ ├── logger.py
│ └── exception.py
├── templates/ # HTML files for Flask app
│ ├── home.html
│ └── index.html
├── app.py # Flask API
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md



---

## ✅ **Dataset Information**
- **Source:** Kaggle - Student Performance Dataset
- **Features:**
  - Gender
  - Race/Ethnicity
  - Parental Level of Education
  - Lunch
  - Test Preparation Course
- **Target:** Student's math score (continuous variable)

---

## ✅ **How to Run the Project**
### **1. Clone the repository**

git clone https://github.com/RukmoddinPatel/Student-Performan-Prediction
cd student-performance-prediction

### 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate


### 3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt



### 4. Train the model
bash
Copy
Edit
python src/pipeline/train_pipeline.py


### 5. Run Flask app
bash
Copy
Edit
python app.py
Visit http://127.0.0.1:5000 in your browser.


### ✅ Results
Best Model: CatBoost Regressor

R² Score: 0.88 on test data

RMSE: ~6.2


### ✅ Future Improvements
Deploy on AWS/GCP using CI/CD pipeline

Hyperparameter tuning for better accuracy

Add explainability (SHAP, LIME)

### ✅ Author
[Rukmoddin Patel] – [LinkedIn](https://www.linkedin.com/in/rukmoddin-patel-a45132260/) | [GitHub](https://github.com/RukmoddinPatel)



```