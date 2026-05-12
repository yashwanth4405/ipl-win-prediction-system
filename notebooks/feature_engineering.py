import pandas as pd

# Load datasets
matches = pd.read_csv("data/matches.csv")
deliveries = pd.read_csv("data/deliveries.csv")

# Merge datasets
final_df = matches.merge(
    deliveries,
    left_on='id',
    right_on='match_id'
)
# Current score calculation

final_df['current_score'] = final_df.groupby('match_id')['total_runs'].cumsum()

print(final_df[['match_id', 'total_runs', 'current_score']].head(10))

# Balls played

# Balls played

final_df['balls_played'] = (
    (final_df['over'] - 1) * 6
) + final_df['ball']

# Balls left

final_df['balls_left'] = 120 - final_df['balls_played']
# Runs left

final_df['runs_left'] = (
    final_df['target_runs'] - final_df['current_score']
)
# Wickets fallen

final_df['wickets_fallen'] = final_df.groupby('match_id')['is_wicket'].cumsum()

# Wickets left

final_df['wickets_left'] = 10 - final_df['wickets_fallen']

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

# Replace infinity values
final_df.replace([float('inf'), -float('inf')], 0, inplace=True)

print(final_df[[
    'current_score',
    'balls_played',
    'current_run_rate',
    'runs_left',
    'balls_left',
    'required_run_rate'
]].head(20))

# Result column

final_df['result'] = (
    final_df['batting_team'] == final_df['winner']
).astype(int)

print(final_df[['batting_team', 'winner', 'result']].head(20))

# Prevent negative values
final_df['wickets_left'] = final_df['wickets_left'].clip(lower=0)

print(final_df[['is_wicket', 'wickets_fallen', 'wickets_left']].head(20))

# Prevent negative values
final_df['runs_left'] = final_df['runs_left'].clip(lower=0)

print(final_df[['target_runs', 'current_score', 'runs_left']].head(15))

# Remove negative values
final_df['balls_left'] = final_df['balls_left'].clip(lower=0)

print(final_df[['over', 'ball', 'balls_played', 'balls_left']].head(10))
# Show merged data
print(final_df.head())

# Shape
print("\nMerged Dataset Shape:")
print(final_df.shape)