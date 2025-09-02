import streamlit as st
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression

st.title("🎈 Obesity Risk Estimator")
st.write("Enter your details to estimate obesity risk.")

# Dummy trained model
model = LinearRegression()
model.fit(np.random.rand(100, 6), np.random.randint(2, size=100))

# Categories
age_groups = ['20-29', '30-39', '40-49', '50-59', '60+']
genders = ['Male', 'Female']
activity_levels = ['Low', 'Medium', 'High']
diets = ['Poor', 'Average', 'Healthy']

# Encoders
age_encoder = OrdinalEncoder(categories=[age_groups])
gender_encoder = OrdinalEncoder(categories=[genders])
activity_encoder = OrdinalEncoder(categories=[activity_levels])
diet_encoder = OrdinalEncoder(categories=[diets])

age_encoder.fit(np.array(age_groups).reshape(-1, 1))
gender_encoder.fit(np.array(genders).reshape(-1, 1))
activity_encoder.fit(np.array(activity_levels).reshape(-1, 1))
diet_encoder.fit(np.array(diets).reshape(-1, 1))

# Streamlit inputs
age_group = st.selectbox("Age Group", age_groups)
gender = st.radio("Gender", genders)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
activity = st.selectbox("Physical Activity", activity_levels)
diet = st.selectbox("Dietary Habits", diets)

if st.button("Predict"):
    # Encode inputs
    features = np.array([
        age_encoder.transform([[age_group]])[0][0],
        gender_encoder.transform([[gender]])[0][0],
        height,
        weight,
        activity_encoder.transform([[activity]])[0][0],
        diet_encoder.transform([[diet]])[0][0]
    ]).reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]

    # Show result
    st.success(f"Predicted obesity risk score: {prediction:.2f}")
    if prediction < 0.4:
        st.markdown("👉 **Low Risk** (maintain your healthy lifestyle!)")
    elif prediction < 0.7:
        st.markdown("👉 **Moderate Risk** (consider improving diet or activity).")
    else:
        st.markdown("👉 **High Risk** (consult a doctor or nutritionist).")
