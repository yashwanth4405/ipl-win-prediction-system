import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
matches = pd.read_csv("data/matches.csv")

# Count wins
wins = matches['winner'].value_counts()

print(wins)

# Plot graph
wins.plot(kind='bar', figsize=(10,5))


plt.title("IPL Team Wins")
plt.xlabel("Teams")
plt.ylabel("Wins")


# Toss impact analysis

toss_win_match_win = matches[matches['toss_winner'] == matches['winner']]

percentage = (len(toss_win_match_win) / len(matches)) * 100

print("\nToss Winner Also Won Match:")
print(round(percentage, 2), "%")

# Load deliveries dataset
deliveries = pd.read_csv("data/deliveries.csv")

# Top run scorers
top_batters = deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(10)

print("\nTop Run Scorers:")
print(top_batters)


top_batters.plot(kind='bar', figsize=(10,5))

plt.title("Top 10 IPL Run Scorers")
plt.xlabel("Players")
plt.ylabel("Runs")

# Top wicket takers

top_bowlers = deliveries[deliveries['is_wicket'] == 1]

top_bowlers = top_bowlers.groupby('bowler')['is_wicket'].sum().sort_values(ascending=False).head(10)

print("\nTop Wicket Takers:")
print(top_bowlers)
top_bowlers.plot(kind='bar', figsize=(10,5))

plt.title("Top 10 Wicket Takers")
plt.xlabel("Bowlers")
plt.ylabel("Wickets")

# Venue analysis

venue_scores = matches.groupby('venue')['target_runs'].mean().sort_values(ascending=False).head(10)

print("\nTop High Scoring Venues:")
print(venue_scores)
venue_scores.plot(kind='bar', figsize=(12,5))

plt.title("Highest Scoring IPL Venues")
plt.xlabel("Venue")
plt.ylabel("Average Target Score")

plt.show()

plt.show()