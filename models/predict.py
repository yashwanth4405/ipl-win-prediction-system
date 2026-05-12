import pickle
import numpy as np
import pandas as pd 

# Load saved model
model = pickle.load(open('models/ipl_model.pkl', 'rb'))

# Example match situation

runs_left = 20
balls_left = 30
wickets_left = 8
current_run_rate = 9
required_run_rate = 4



input_data = pd.DataFrame([{
    'runs_left': runs_left,
    'balls_left': balls_left,
    'wickets_left': wickets_left,
    'current_run_rate': current_run_rate,
    'required_run_rate': required_run_rate
}])

# Predict probability

probability = model.predict_proba(input_data)

# Winning probability

win_probability = probability[0][1] * 100

print("\nWinning Probability:")
print(round(win_probability, 2), "%")