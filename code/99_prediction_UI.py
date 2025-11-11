# save this as run_matchup_gui.py
import streamlit as st
import joblib
import pandas as pd
from io import StringIO
import requests

# --------------------------
# Load models
# --------------------------
rf = joblib.load("../models/rf_matchup_model.pkl")
xgb = joblib.load("../models/xgb_matchup_model.pkl")


# --------------------------
# Load future games
# --------------------------
tbd_games = pd.read_csv(
    "https://raw.githubusercontent.com/jake-lukasik/Capstone-Prediction-App/refs/heads/main/datasets/datasets/all-possible-matchups.csv"
)


# --------------------------
# Helper: process matchup
# --------------------------
def get_matchup_for_prediction(future_df, home_team, away_team):
    mask = (
        ((future_df["Team_A"] == home_team) & (future_df["Team_B"] == away_team)) |
        ((future_df["Team_A"] == away_team) & (future_df["Team_B"] == home_team))
    )
    
    if mask.sum() == 0:
        raise ValueError(f"No matchup found for {home_team} vs {away_team}")
    
    row = future_df[mask].copy().iloc[0]
    
    if row["Team_A"] != home_team:
        # Swap teams
        row["Team_A"], row["Team_B"] = row["Team_B"], row["Team_A"]
        # Flip stat diffs
        stat_cols = [c for c in row.index if "_diff" in c]
        row[stat_cols] = -row[stat_cols]
    
    if "result" in row.index:
        row = row.drop("result")
        
    return row.to_frame().T


# --------------------------
# Helper: Build LLM Prompt
# --------------------------
def build_nfl_prompt(processed_matchup: pd.DataFrame) -> str:
    """
    Build an LLM-ready prompt from a single-row matchup DataFrame.
    """
    row = processed_matchup.iloc[0]
    team_a = row["Team_A"]
    team_b = row["Team_B"]

    features = row.drop(["Team_A", "Team_B"], errors="ignore")
    feature_text = "\n".join(
        [f"{k}: {v}" for k, v in features.items() if pd.notnull(v)]
    )

    prompt = f"""

    You are an expert NFL data analyst predicting game outcomes using comparative team statistics.

    Each record compares two NFL teams' recent performance. 
    All numeric features represent the difference: (Team_A - Team_B).
    Positive values favor Team_A; negative values favor Team_B.
    Along with this, Team_A is always the home team.

    Using the provided data, predict which team is more likely to win.
    Base your reasoning on key factors like offensive efficiency (Pts_diff, Y/P_y_diff), 
    defensive strength (Pts_allowed_diff, Y/P_allowed_diff), and turnover metrics (TO%_diff, Takeaways_diff).

    Return your reasoning in 2–3 sentences, followed by your prediction in the exact format:
    "Prediction: Team_A wins" or "Prediction: Team_B wins" where Team A or B is replaced with the actual team name.

    In another paragraph, 4-5 sentences, give some insight as to how you think that the game will go. Will it 
    be an offensive shootout? A defensive masterclass? Lastly, give a range as to how much you think 
    the winning team will win by, similar to a sports betting spread

    Game Data:
    Team_A: {team_a}
    Team_B: {team_b}

    Features:
    {feature_text}
    """
    return prompt.strip()


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="NFL Matchup Predictor", page_icon="🏈", layout="wide")
st.title("🏈 NFL Matchup Predictor")

# External Links
st.markdown(
    """
    ### 🔗 Quick Links  
    [💬 ChatGPT](https://chat.openai.com) | 
    [🌐 Gemini](https://gemini.google.com) | 
    [🤖 Copilot](https://copilot.microsoft.com)
    ---
    """
)

# Dropdowns for teams
teams = sorted(list(set(tbd_games["Team_A"]).union(set(tbd_games["Team_B"]))))
home_team = st.selectbox("Select Home Team", teams)
away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team])

if st.button("Predict Winner"):
    try:
        processed_matchup = get_matchup_for_prediction(tbd_games, home_team, away_team)
        
        X_new = processed_matchup.drop(columns=["Team_A", "Team_B"])
        X_new = X_new.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        rf_result = rf.predict(X_new)[0]
        xgb_result = xgb.predict(X_new)[0]
        
        game_prediction = pd.DataFrame({
            "Home Team": [home_team],
            "Away Team": [away_team],
            "result_RF": [rf_result],
            "result_XGB": [xgb_result],
            "Winner_RF": [home_team if rf_result == 1 else away_team],
            "Winner_XGB": [home_team if xgb_result == 1 else away_team]
        })
        
        st.subheader("Prediction Results")
        st.dataframe(game_prediction)

        # Automatically build and show LLM prompt
        llm_prompt = build_nfl_prompt(processed_matchup)
        st.subheader("🧠 LLM Prompt (Auto-Generated)")
        st.text_area("Generated Prompt", llm_prompt, height=400)
        st.info("✅ You can copy this prompt and paste it into ChatGPT, Gemini, or Copilot!")

    except Exception as e:
        st.error(str(e))