import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# Load datasets
# -----------------------------

matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# -----------------------------
# Merge datasets
# -----------------------------

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
final_df['current_score'] = final_df.groupby(
    'match_id'
)['total_runs'].cumsum()

# Balls played
valid_mask = (
    (final_df['over'] >= 1) &
    (final_df['ball'] >= 1)
)

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
final_df['wickets_fallen'] = final_df.groupby(
    'match_id'
)['is_wicket'].cumsum()

# Wickets left
final_df['wickets_left'] = 10 - final_df['wickets_fallen']
final_df['wickets_left'] = final_df['wickets_left'].clip(lower=0)

# Required Run Rate
final_df['required_run_rate'] = (
    (final_df['runs_left'] * 6)
    / final_df['balls_left']
)

# Remove infinity values
final_df.replace(
    [float('inf'), -float('inf')],
    0,
    inplace=True
)

# -----------------------------
# Remove unrealistic situations
# -----------------------------

final_df = final_df[
    (final_df['balls_left'] > 12) &
    (final_df['runs_left'] > 0)
]

# -----------------------------
# Result Column
# -----------------------------

final_df['result'] = (
    final_df['batting_team'] == final_df['winner']
).astype(int)

# -----------------------------
# Create Bowling Team
# -----------------------------

final_df['bowling_team'] = final_df['team1']

final_df.loc[
    final_df['batting_team'] == final_df['team1'],
    'bowling_team'
] = final_df['team2']

# -----------------------------
# Keep Required Columns
# -----------------------------

final_df = final_df[[
    'match_id',
    'batting_team',
    'bowling_team',
    'runs_left',
    'balls_left',
    'wickets_left',
    'required_run_rate',
    'result'
]]

# Remove missing values
final_df = final_df.dropna()

# -----------------------------
# Split By Match ID
# -----------------------------

match_ids = final_df['match_id'].unique()

train_match_ids, test_match_ids = train_test_split(
    match_ids,
    test_size=0.2,
    random_state=42
)

train_df = final_df[
    final_df['match_id'].isin(train_match_ids)
]

test_df = final_df[
    final_df['match_id'].isin(test_match_ids)
]

# -----------------------------
# Features and Target
# -----------------------------

X_train = train_df[[
    'batting_team',
    'bowling_team',
    'runs_left',
    'balls_left',
    'wickets_left',
    'required_run_rate'
]]

y_train = train_df['result']

X_test = test_df[[
    'batting_team',
    'bowling_team',
    'runs_left',
    'balls_left',
    'wickets_left',
    'required_run_rate'
]]

y_test = test_df['result']

# -----------------------------
# One Hot Encoding
# -----------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            'team_encoder',
            OneHotEncoder(handle_unknown='ignore'),
            ['batting_team', 'bowling_team']
        )
    ],
    remainder='passthrough'
)

# -----------------------------
# Model Pipeline
# -----------------------------

model = Pipeline([
    (
        'preprocessor',
        preprocessor
    ),
    (
        'classifier',
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

# -----------------------------
# Train Model
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, predictions)

# -----------------------------
# Save Model
# -----------------------------

pickle.dump(
    model,
    open('models/random_forest_model.pkl', 'wb')
)

print("\nModel Saved Successfully")

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")