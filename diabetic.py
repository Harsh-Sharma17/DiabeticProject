# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("diabetics.csv")

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=30)

# Train Logistic Regression model directly
model = LogisticRegression(max_iter=1000)  # Increase iterations for convergence
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
xticklabels=["No Diabetes", "Diabetes"], 
yticklabels=["No Diabetes", "Diabetes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

outcome_counts = df['Outcome'].value_counts()
plt.figure(figsize=(6,4))
plt.bar(['No Diabetes (0)', 'Diabetes (1)'], outcome_counts, color=['green', 'red'])
plt.title('Diabetes Outcome Distribution')
plt.xlabel('Outcome')
plt.ylabel('Number of Patients')
plt.show()


# Save model (no scaler needed)
joblib.dump(model, "diabetes_model.pkl")


# app.py
import streamlit as st
import numpy as np
import joblib
# Load the logistic regression model (no scaler now)
model = joblib.load("diabetes_model.pkl")

st.title(" Diabetes Prediction App")
st.write("Enter the details below to check diabetes risk.")

# Input fields
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0.0)
bp = st.number_input("Blood Pressure", min_value=0.0)
skin = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Predict using logistic regression
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.write(f"Probability of being diabetic:")
    
    if prediction[0] == 1:
        st.error("Prediction: Diabetic")
    else:

        st.success("Prediction: Not Diabetic")
