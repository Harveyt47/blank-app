import streamlit as streamlit
import pandas as pd
import plotly.graph_objects as go
from Match_Modules import PlayerSimulator, MatchSimulator  # Adjust import based on your file structure
from dotenv import load_dotenv
import os
import numpy as np

# Title and sidebar
streamlit.title("Match Simulation Dashboard")
streamlit.sidebar.header("Player Selection")

# Load player data (replace with your actual data source)
@streamlit.cache_data
def load_player_data():
    # Example: Load your DataFrame (adjust path or method)
    file = r'../../data_transformations/latest_database/160525_player_long_dups_sports_radar.csv'
    df = pd.read_csv(file, low_memory=False)
    unique_players = df['player_name'].dropna().unique().tolist()
    return unique_players

player_options = load_player_data()
player1 = streamlit.sidebar.selectbox("Select Player ", player_options, index=0)
player2 = streamlit.sidebar.selectbox("Select Opponent", player_options, index=1 if len(player_options) > 1 else 0)

# Simulation button
if streamlit.sidebar.button("Run Simulation"):
    with streamlit.spinner("Simulating..."):
        # Load player simulators from match data
        player1_sim = PlayerSimulator.from_match_data(player1, df=pd.read_csv("your_data.csv"), surface="Hard")
        player2_sim = PlayerSimulator.from_match_data(player2, df=pd.read_csv("your_data.csv"), surface="Hard")

        # Initialize and run simulation
        match_sim = MatchSimulator(player1_sim, player2_sim, num_sets_to_win=2, rounds='R1')
        match_results, df = match_sim.simulate_many_matches_parallel(n_simulations=1000, verbose=0)
        # Display the table
        streamlit.subheader("Match Statistics Averages")
        streamlit.dataframe(df)  # Display the raw DataFrame (styling from style object won't render)

        # Display additional charts from visualize_results
        match_sim.visualize_results(match_results)

# Instructions
streamlit.markdown("Select two players from the sidebar and click 'Run Simulation' to see the win probability chart and other statistics.")
