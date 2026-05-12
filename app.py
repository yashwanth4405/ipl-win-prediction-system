import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load Trained Model
# -----------------------------

model = pickle.load(
    open('models/random_forest_model.pkl', 'rb')
)

# -----------------------------
# App Title
# -----------------------------

st.title("IPL Win Probability Predictor")

st.write("Predict IPL Match Winning Chances")

# -----------------------------
# Teams
# -----------------------------

teams = [
    'Kolkata Knight Riders',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kings XI Punjab',
    'Delhi Daredevils',
    'Sunrisers Hyderabad'
]

# -----------------------------
# Team Selection
# -----------------------------

batting_team = st.selectbox(
    'Batting Team',
    sorted(teams)
)

bowling_team = st.selectbox(
    'Bowling Team',
    sorted(teams)
)

# -----------------------------
# Match Inputs
# -----------------------------

runs_left = st.number_input(
    'Runs Left',
    min_value=0
)

balls_left = st.number_input(
    'Balls Left',
    min_value=0,
    max_value=120
)

wickets_left = st.number_input(
    'Wickets Left',
    min_value=0,
    max_value=10
)

current_run_rate = st.number_input(
    'Current Run Rate',
    min_value=0.0
)

required_run_rate = st.number_input(
    'Required Run Rate',
    min_value=0.0
)

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Winning Probability"):

    # Prevent same teams

    if batting_team == bowling_team:
        st.error("Batting Team and Bowling Team cannot be same.")

    else:

        # Input dataframe

        input_data = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'runs_left': runs_left,
            'balls_left': balls_left,
            'wickets_left': wickets_left,
            'current_run_rate': current_run_rate,
            'required_run_rate': required_run_rate
        }])

        # Prediction probability

        probability = model.predict_proba(input_data)

        win_probability = probability[0][1] * 100
        lose_probability = probability[0][0] * 100

        # Show probabilities

        st.success(
            f"Winning Probability: {round(win_probability, 2)}%"
        )

        st.error(
            f"Losing Probability: {round(lose_probability, 2)}%"
        )

        # Final Winner Prediction

        if win_probability > 50:

            st.header(
                f"Predicted Winner: {batting_team}"
            )

        else:

            st.header(
                f"Predicted Winner: {bowling_team}"
            )