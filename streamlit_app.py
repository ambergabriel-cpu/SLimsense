import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
None selected 

import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression  # Changed to LinearRegression

# Let's assume the model is already trained (as from your previous steps)
# Here we simulate that with some dummy model
model = LinearRegression()  # Changed to LinearRegression
model.fit(np.random.rand(100, 6), np.random.randint(2, size=100))  # Dummy training

# Collect user input
def collect_user_input():
    print("Please enter the following details:")

    age_group = input("Age Group (e.g., '20-29', '30-39', '40-49', '50-59', '60+'): ")
    gender = input("Gender ('Male' or 'Female'): ")
    height = int(input("Height in cm: "))
    weight = int(input("Weight in kg: "))
    physical_activity = input("Physical Activity ('Low', 'Medium', 'High'): ")
    dietary_habits = input("Dietary Habits ('Poor', 'Average', 'Healthy'): ")

    return [age_group, gender, height, weight, physical_activity, dietary_habits]

# Function to preprocess input
def preprocess_input(user_input):
    # Define categories and encoders
    age_groups = ['20-29', '30-39', '40-49', '50-59', '60+']
    genders = ['Male', 'Female']
    activity_levels = ['Low', 'Medium', 'High']
    diets = ['Poor', 'Average', 'Healthy']

    # Initialize encoders
    age_encoder = OrdinalEncoder(categories=[age_groups])
    gender_encoder = OrdinalEncoder(categories=[genders])
    activity_encoder = OrdinalEncoder(categories=[activity_levels])
    diet_encoder = OrdinalEncoder(categories=[diets])

    # Encode the user input
    age_encoded = age_encoder.fit_transform(np.array(user_input[0]).reshape(-1, 1))
    gender_encoded = gender_encoder.fit_transform(np.array(user_input[1]).reshape(-1, 1))
    activity_encoded = activity_encoder.fit_transform(np.array(user_input[4]).reshape(-1, 1))
    diet_encoded = diet_encoder.fit_transform(np.array(user_input[5]).reshape(-1, 1))

    # Prepare the features array
    features = [
        age_encoded[0][0],  # Age group encoding
        gender_encoded[0][0],  # Gender encoding
        user_input[2],  # Height in cm
        user_input[3],  # Weight in kg
        activity_encoded[0][0],  # Activity level encoding
        diet_encoded[0][0],  # Diet encoding
    ]
    
    return np.array(features).reshape(1, -1)

# Collect the user input
user_input = collect_user_input()

# Preprocess the input
processed_input = preprocess_input(user_input)

# Make a prediction
prediction = model.predict(processed_input)

# Output the result
print(f"The model predicts a value of {prediction[0]:.2f} for obesity risk (higher values indicate higher risk).")

