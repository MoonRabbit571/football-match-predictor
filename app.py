import streamlit as st
import pandas as pd
import pickle

# Load model
with open("match_outcome_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load teams
teams_df = pd.read_csv("results.csv")[['home_team', 'away_team']]
all_teams = pd.unique(teams_df[['home_team', 'away_team']].values.ravel())

st.title("⚽ Football Match Outcome Predictor")

home_team = st.selectbox("Select Home Team", sorted(all_teams))
away_team = st.selectbox("Select Away Team", sorted(all_teams))

if st.button("Predict Outcome"):
    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        home_code = pd.Categorical([home_team], categories=all_teams).codes[0]
        away_code = pd.Categorical([away_team], categories=all_teams).codes[0]

        prediction = model.predict([[home_code, away_code]])[0]
        outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        st.success(f"🏁 Predicted Outcome: **{outcome_map[prediction]}**")
