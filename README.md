# IPL Win Probability Prediction System 🏏📊

## Overview

This project predicts the winning probability of IPL teams during live match situations using Machine Learning.

The system analyzes match conditions such as:
- Runs left
- Balls left
- Wickets left
- Current Run Rate
- Required Run Rate
- Batting Team
- Bowling Team

and predicts:
- Winning Probability
- Losing Probability
- Predicted Winner

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
- Git & GitHub

---

## Machine Learning Models Used

- Logistic Regression
- Random Forest
- XGBoost

After comparing all models, Random Forest produced the most stable and realistic predictions.

---

## Features

✅ Feature Engineering  
✅ Team Encoding  
✅ Match-wise Data Splitting  
✅ Data Leakage Handling  
✅ Real-time Prediction System  
✅ Interactive Streamlit Web App  

---

## Project Structure

```text
ipl sports analytics/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
├── models/
├── notebooks/


Requirements are in Seperate file 
  check it out!!

How to run Run

Install Dependencies 
   pip install -r requirements.txt

Run Application 
   streamlit run app.py


Author 
Yashwanth S
