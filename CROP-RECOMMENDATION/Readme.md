# 🌾 Crop & Irrigation-method Recommendation System
### Machine Learning–Based Smart Crop and Irrigation Recommendation

---

## 📌 Overview

The Crop Recommendation System is a machine learning–based decision support application designed to recommend the **most suitable crop** for cultivation based on soil nutrients and environmental conditions. The system analyzes key agricultural parameters such as **NPK values, temperature, humidity, rainfall, wind speed, and soil pH** to predict the optimal crop using supervised learning techniques.

Multiple classification models were trained and evaluated to identify the best-performing algorithm. Based on comparative analysis, the **Random Forest classifier** achieved the highest accuracy and was selected for deployment. The system is integrated with a **Streamlit-based interface** that enables real-time predictions and also provides **irrigation method recommendations** for the predicted crop, promoting precision agriculture and resource efficiency.

---

## 🎯 Objectives

- Recommend the most suitable crop based on soil and environmental parameters  
- Compare multiple machine learning models and select the best-performing one  
- Provide real-time crop prediction through an interactive web interface  
- Suggest appropriate irrigation methods for the predicted crop  
---

## 📂 Data Source

The system uses the **Crop_Recommendation.csv** dataset, which contains agricultural and environmental features relevant to crop suitability.

### Dataset Attributes
- **Soil Nutrients:** Nitrogen (N), Phosphorus (P), Potassium (K)  
- **Environmental Parameters:** Temperature (°C), Humidity (%), Rainfall (mm), Wind Speed  
- **Soil Property:** pH level  
- **Target Variable:** Crop label (categorical) And irrigation method

### Data Characteristics
- No missing values  
- No duplicate records  
- Multiple crop classes representing diverse agricultural conditions  

---

## 🧠 Machine Learning Approach

### Models Implemented
The following machine learning classification algorithms were trained and evaluated:

- Decision Tree  
- Gaussian Naive Bayes  
- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
- Random Forest  
- XGBoost  

### Data Preprocessing
- Feature–target separation  
- Label encoding of crop names  
- Train–test split (80:20 ratio)  
- Reproducibility ensured using fixed random state  

---

## 📊 Model Evaluation & Selection

Model performance was evaluated using:

- Accuracy score  
- Classification report (precision, recall, F1-score)  
- Cross-validation  
- Confusion matrix analysis  
- Feature importance visualization  

### Best Model
- **Random Forest Classifier**
  - Achieved highest overall accuracy (~99.5%)
  - Demonstrated strong generalization across crop classes
  - Provided interpretability through feature importance

The selected model was serialized using **Pickle** for deployment.

---

## 🔄 System Workflow

1. User inputs soil and environmental parameters  
2. Input data is processed and passed to the trained Random Forest model  
3. The model predicts the most suitable crop  
4. Based on the predicted crop, an irrigation method is suggested  
5. Results are displayed via the Streamlit interface  

---

## 🖥️ Application Interface

- Built using **Streamlit**
- User-friendly form-based input
- Instant prediction results
- Clear display of:
  - Recommended crop
  - Suggested irrigation method  

The interface is designed for accessibility and ease of use by non-technical users.

---

## 🌱 Irrigation Recommendation Module

The system includes a rule-based irrigation advisory component that suggests suitable irrigation techniques based on crop type, such as:

- Drip irrigation  
- Sprinkler irrigation  
- Furrow irrigation  
- Flood irrigation  

This feature helps farmers optimize water usage and supports sustainable irrigation practices.

---

## 📈 Results & Analysis

- Random Forest outperformed all other models in accuracy and stability  
- Feature importance analysis showed **rainfall, soil pH, and NPK values** as dominant factors  
- Per-crop accuracy remained consistently high across most classes  
- Confusion matrix confirmed minimal misclassification  

The results validate the reliability and effectiveness of the system for real-world crop planning.

---

## 🧪 Tech Stack

### Core Technologies
- Python  
- Scikit-learn  
- Random Forest, XGBoost  
- Pandas, NumPy  

### Visualization & Deployment
- Matplotlib, Seaborn  
- Streamlit  
 
---

## 🚀 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/crop-recommendation-system.git

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 🔮 Future Enhancements
- Integration with real-time weather APIs
- Mobile application deployment
- Multilingual user interface
