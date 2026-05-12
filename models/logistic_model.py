import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load engineered dataset
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# Merge datasets
final_df = matches.merge(
    deliveries,
    left_on='id',
    right_on='match_id'
)
# Keep only second innings

final_df = final_df[final_df['inning'] == 2]

# -----------------------------
# Feature Engineering
# -----------------------------

# Current score
final_df['current_score'] = final_df.groupby('match_id')['total_runs'].cumsum()

# Balls played
valid_mask = (final_df['over'] >= 1) & (final_df['ball'] >= 1)

final_df.loc[valid_mask, 'balls_played'] = (
    ((final_df.loc[valid_mask, 'over'] - 1) * 6)
    + (final_df.loc[valid_mask, 'ball'] - 1)
)

# Balls left
final_df['balls_left'] = 120 - final_df['balls_played']
final_df['balls_left'] = final_df['balls_left'].clip(lower=0)

# Runs left
final_df['runs_left'] = (
    final_df['target_runs'] - final_df['current_score']
)
final_df['runs_left'] = final_df['runs_left'].clip(lower=0)

# Wickets fallen
final_df['wickets_fallen'] = final_df.groupby('match_id')['is_wicket'].cumsum()

# Wickets left
final_df['wickets_left'] = 10 - final_df['wickets_fallen']
final_df['wickets_left'] = final_df['wickets_left'].clip(lower=0)

# Current Run Rate
final_df['current_run_rate'] = (
    (final_df['current_score'] * 6)
    / final_df['balls_played']
)

# Required Run Rate
final_df['required_run_rate'] = (
    (final_df['runs_left'] * 6)
    / final_df['balls_left']
)

# Remove infinity
final_df.replace([float('inf'), -float('inf')], 0, inplace=True)

# Result column
final_df['result'] = (
    final_df['batting_team'] == final_df['winner']
).astype(int)

# -----------------------------
# ML Features
# -----------------------------

# Keep only required columns

final_df = final_df[[
    'runs_left',
    'balls_left',
    'wickets_left',
    'current_run_rate',
    'required_run_rate',
    'result'
]]

# Remove missing values
final_df = final_df.dropna()
# Features

X = final_df[[
    'runs_left',
    'balls_left',
    'wickets_left',
    'current_run_rate',
    'required_run_rate'
]]

# Target

y = final_df['result']
# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model Training
# -----------------------------

model = LogisticRegression()

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

predictions = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

# Save model
pickle.dump(model, open('models/ipl_model.pkl', 'wb'))

print("\nModel Saved Successfully")

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")