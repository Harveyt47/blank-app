import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict, Counter
import multiprocessing
from joblib import Parallel, delayed
import copy
from scipy.stats import beta, gaussian_kde
from IPython.display import display
from statsmodels.stats.proportion import proportion_confint
import streamlit as st  # Import Streamlit inside the method

# PlayerSimulator class (unchanged)
class PlayerSimulator:
    def __init__(self, name: None, beta_params: dict, adjuster_models: dict = None, defaulted=False):
        self.beta_params = beta_params
        self.name = name
        self.adjuster_models = adjuster_models or {}
        self.defaulted = defaulted

    @classmethod
    def from_match_data(cls, player_name, df, surface, no_matches=2000):
        name = player_name
        player_data = df[(df['player_name'] == player_name) & (df['surface'] == surface)]
        player_data = player_data[-no_matches:]
        beta_params = {}
        metrics = [
            'player_perc_1stIn',
            'player_perc_ace_1st',
            'player_perc_rally_1st',
            'player_perc_2ndIn',
            'player_perc_ace_2nd',
            'player_perc_rally_2nd',
            'player_perc_return_rally_1st',
            'player_perc_return_rally_2nd',
            'player_perc_bp_saved',
            'player_perc_bp_won_returner'
        ]
        defaulted = False
        for metric in metrics:
            if metric in player_data.columns:
                metric_data = player_data[metric].dropna()
                if len(metric_data) >= 5:
                    alpha, beta_, loc, scale = beta.fit(np.clip(metric_data.values, 0.0001, 0.9999), floc=0, fscale=1)
                    beta_params[metric] = (alpha, beta_)
                    defaulted = False
                else:
                    beta_params[metric] = (np.nan, np.nan)
                    defaulted = True
            else:
                beta_params[metric] = (np.nan, np.nan)
                defaulted = True
        return cls(name, beta_params, defaulted=defaulted)

    def sample_metric(self, metric_name, sample=1):

        alpha, beta_ = self.beta_params.get(metric_name, (np.nan, np.nan))
        if np.isnan(alpha) or np.isnan(beta_):
            return 0.5
        if sample == 1:
            base_sample = beta.rvs(alpha, beta_)
        else:
            base_sample = alpha / (alpha + beta_)
        return base_sample

    def sample_all_metrics(self, sample=1):
        return {metric: self.sample_metric(metric, sample) for metric in self.beta_params.keys()}

    def get_alpha_beta(self, stat_name):
        alpha, beta_ = self.beta_params.get(stat_name, (np.nan, np.nan))
        return alpha, beta_

# MatchSimulator class (updated with new stats and KDE charts)
class MatchSimulator:
    def __init__(self, player_sim: PlayerSimulator, opponent_sim: PlayerSimulator, num_sets_to_win=2, rounds='R1', sample_point=1):
        self.player_sim = player_sim
        self.opponent_sim = opponent_sim
        self.num_sets_to_win = num_sets_to_win
        self.player_name = player_sim.name
        self.opponent_name = opponent_sim.name
        self.rounds = rounds
        self.sample_point = sample_point
        self.stats_to_track = [
            '1st_ace', '1st_in', '1st_rally',
            '2nd_ace', '2nd_in', '2nd_rally',
            'serve_attempt', 'first_serve_in',
            'ace', 'double_fault', 'rally_won',
            'bp_faced', 'bp_saved', 'bp_won',
            'first_serve_won', 'second_serve_won',
            'first_serve_faced', 'first_serve_won_return',
            'second_serve_faced', 'second_serve_won_return',
            'serve_point_won', 'return_point_won',
            'total_points_won'
        ]
        self.reset_stat_outcomes()
        metrics = [
            'player_perc_1stIn',
            'player_perc_ace_1st',
            'player_perc_rally_1st',
            'player_perc_2ndIn',
            'player_perc_ace_2nd',
            'player_perc_rally_2nd',
            'player_perc_return_rally_1st',
            'player_perc_return_rally_2nd',
            'player_perc_bp_saved',
            'player_perc_bp_won_returner'
        ]
        self.alpha_beta_log = {
            'player': {stat: self.player_sim.get_alpha_beta(stat) for stat in metrics},
            'opponent': {stat: self.opponent_sim.get_alpha_beta(stat) for stat in metrics}
        }

    def reset_stat_outcomes(self):
        self.stat_outcomes = {
            'player': {stat: {'success': 0, 'fail': 0} for stat in self.stats_to_track},
            'opponent': {stat: {'success': 0, 'fail': 0} for stat in self.stats_to_track}
        }

    def log_stat(self, player_id, stat, outcome):
        self.stat_outcomes[player_id][stat][outcome] += 1

    def simulate_point(self, server_sim: PlayerSimulator, returner_sim: PlayerSimulator, is_break_point=False):
        skills_server = server_sim.sample_all_metrics(sample=self.sample_point)
        skills_returner = returner_sim.sample_all_metrics(sample=self.sample_point)
        server_id = 'player' if server_sim == self.player_sim else 'opponent'
        returner_id = 'opponent' if server_id == 'player' else 'player'
        self.log_stat(server_id, 'serve_attempt', 'success')
        first_in = np.random.rand() < skills_server['player_perc_1stIn']
        self.log_stat(server_id, '1st_in', 'success' if first_in else 'fail')
        self.log_stat(returner_id, 'first_serve_faced', 'success')
        if first_in:
            self.log_stat(server_id, 'first_serve_in', 'success')
            first_ace = np.random.rand() < skills_server['player_perc_ace_1st']
            self.log_stat(server_id, '1st_ace', 'success' if first_ace else 'fail')
            if first_ace:
                self.log_stat(server_id, 'ace', 'success')
                self.log_stat(server_id, 'first_serve_won', 'success')
                self.log_stat(server_id, 'serve_point_won', 'success')
                self.log_stat(server_id, 'total_points_won', 'success')
                return 1
            else:
                prob_win_rally_1st = skills_server['player_perc_rally_1st'] / (
                skills_server['player_perc_rally_1st'] + skills_returner['player_perc_return_rally_1st']
                )
                if is_break_point:
                    break_point_prob = skills_server['player_perc_bp_saved'] / (
                        skills_server['player_perc_bp_saved'] + skills_returner['player_perc_bp_won_returner'] + 1e-6
                    )
                    weight = 0.5
                    prob_win_rally_1st = (1 - weight) * prob_win_rally_1st + weight * break_point_prob
                    prob_win_rally_1st = min(max(prob_win_rally_1st, 0.01), 0.99)
                first_rally_success = np.random.rand() < prob_win_rally_1st
                self.log_stat(server_id, '1st_rally', 'success' if first_rally_success else 'fail')
                if first_rally_success:
                    self.log_stat(server_id, 'rally_won', 'success')
                    self.log_stat(server_id, 'first_serve_won', 'success')
                    self.log_stat(server_id, 'serve_point_won', 'success')
                    self.log_stat(server_id, 'total_points_won', 'success')
                    return 1
                else:
                    self.log_stat(returner_id, 'rally_won', 'success')
                    self.log_stat(returner_id, 'first_serve_won_return', 'success')
                    self.log_stat(returner_id, 'return_point_won', 'success')
                    self.log_stat(returner_id, 'total_points_won', 'success')
                    return 0
        else:
            second_in = np.random.rand() < skills_server['player_perc_2ndIn']
            self.log_stat(server_id, '2nd_in', 'success' if second_in else 'fail')
            self.log_stat(returner_id, 'second_serve_faced', 'success')
            if second_in:
                second_ace = np.random.rand() < skills_server['player_perc_ace_2nd']
                self.log_stat(server_id, '2nd_ace', 'success' if second_ace else 'fail')
                if second_ace:
                    self.log_stat(server_id, 'ace', 'success')
                    self.log_stat(server_id, 'second_serve_won', 'success')
                    self.log_stat(server_id, 'serve_point_won', 'success')
                    self.log_stat(server_id, 'total_points_won', 'success')
                    return 1
                else:
                    prob_win_rally_2nd = skills_server['player_perc_rally_2nd'] / (
                        skills_server['player_perc_rally_2nd'] + skills_returner['player_perc_return_rally_2nd']
                    )
                    if is_break_point:
                        break_point_prob = skills_server['player_perc_bp_saved'] / (
                            skills_server['player_perc_bp_saved'] + skills_returner['player_perc_bp_won_returner'] + 1e-6
                        )
                        weight = 0.5
                        prob_win_rally_2nd = (1 - weight) * prob_win_rally_2nd + weight * break_point_prob
                        prob_win_rally_2nd = min(max(prob_win_rally_2nd, 0.01), 0.99)
                    second_rally = np.random.rand() < prob_win_rally_2nd
                    self.log_stat(server_id, '2nd_rally', 'success' if second_rally else 'fail')
                    if second_rally:
                        self.log_stat(server_id, 'rally_won', 'success')
                        self.log_stat(server_id, 'second_serve_won', 'success')
                        self.log_stat(server_id, 'serve_point_won', 'success')
                        self.log_stat(server_id, 'total_points_won', 'success')
                        return 1
                    else:
                        self.log_stat(returner_id, 'rally_won', 'success')
                        self.log_stat(returner_id, 'second_serve_won_return', 'success')
                        self.log_stat(returner_id, 'return_point_won', 'success')
                        self.log_stat(returner_id, 'total_points_won', 'success')
                        return 0
            else:
                self.log_stat(server_id, 'double_fault', 'success')
                self.log_stat(returner_id, 'return_point_won', 'success')
                self.log_stat(returner_id, 'total_points_won', 'success')
                return 0

    def simulate_game(self, server_sim, returner_sim):
        server_points = 0
        returner_points = 0
        points_played = 0
        server_id = 'player' if server_sim == self.player_sim else 'opponent'
        returner_id = 'opponent' if server_id == 'player' else 'player'
        while True:
            points_played += 1
            is_break_point = returner_points >= 3 and server_points < returner_points
            if is_break_point:
                self.log_stat(server_id, 'bp_faced', 'success')
            point_winner = self.simulate_point(server_sim, returner_sim, is_break_point)
            if point_winner == 1:
                server_points += 1
                if is_break_point:
                    self.log_stat(server_id, 'bp_saved', 'success')
            else:
                returner_points += 1
                if is_break_point:
                    self.log_stat(returner_id, 'bp_won', 'success')
            if server_points >= 4 and (server_points - returner_points) >= 2:
                return 1, points_played
            if returner_points >= 4 and (returner_points - server_points) >= 2:
                return 0, points_played

    def simulate_set(self, starting_server=0):
        player_games = 0
        opponent_games = 0
        game_number = 0
        total_points_in_set = 0
        while True:
            current_server = (starting_server + game_number) % 2
            if current_server == 0:
                result, points = self.simulate_game(self.player_sim, self.opponent_sim)
                total_points_in_set += points
                if result == 1:
                    player_games += 1
                else:
                    opponent_games += 1
            else:
                result, points = self.simulate_game(self.opponent_sim, self.player_sim)
                total_points_in_set += points
                if result == 1:
                    opponent_games += 1
                else:
                    player_games += 1
            if player_games >= 6 and player_games - opponent_games >= 2:
                return 'Player', player_games, opponent_games, total_points_in_set
            if opponent_games >= 6 and opponent_games - player_games >= 2:
                return 'Opponent', player_games, opponent_games, total_points_in_set
            if player_games == 6 and opponent_games == 6:
                tiebreak_points = {'player': 0, 'opponent': 0}
                server = starting_server
                point_count = 0
                while abs(tiebreak_points['player'] - tiebreak_points['opponent']) < 2 or max(tiebreak_points.values()) < 7:
                    server_sim = self.player_sim if server == 0 else self.opponent_sim
                    returner_sim = self.opponent_sim if server == 0 else self.player_sim
                    point_winner = self.simulate_point(server_sim, returner_sim)
                    winner_id = 'player' if (point_winner == 1 and server == 0) or (point_winner == 0 and server == 1) else 'opponent'
                    tiebreak_points[winner_id] += 1
                    point_count += 1
                    if point_count % 2 == 0:
                        server = 1 - server
                winner = 'Player' if tiebreak_points['player'] > tiebreak_points['opponent'] else 'Opponent'
                if winner == 'Player':
                    player_games += 1
                else:
                    opponent_games += 1
                total_points_in_set += point_count
                return winner, player_games, opponent_games, total_points_in_set
            game_number += 1

    def simulate_match(self):
        player_sets = 0
        opponent_sets = 0
        starting_server = 0
        self.reset_stat_outcomes()
        match_stats = {
            'player': self.player_name,
            'opponent': self.opponent_name,
            'round': self.rounds,
            'player_sets': 0,
            'opponent_sets': 0,
            'player_games': 0,
            'opponent_games': 0,
            'points_played': 0,
            'aces_player': 0,
            'aces_opponent': 0,
            'double_faults_player': 0,
            'double_faults_opponent': 0,
            'rallies_won_player': 0,
            'rallies_won_opponent': 0,
            'bp_faced_player': 0,
            'bp_faced_opponent': 0,
            'bp_saved_player': 0,
            'bp_saved_opponent': 0,
            'bp_won_player': 0,
            'bp_won_opponent': 0,
            'percent_first_serves_in_player': 0,
            'percent_first_serves_in_opponent': 0,
            'percent_bp_saved_player': 0,
            'percent_bp_saved_opponent': 0,
            'percent_bp_won_player': 0,
            'percent_bp_won_opponent': 0,
            'percent_serve_points_won_player': 0,
            'percent_serve_points_won_opponent': 0,
            'percent_return_points_won_player': 0,
            'percent_return_points_won_opponent': 0,
            'point_dominance_player': 0,
            'point_dominance_opponent': 0,
            'percent_points_won_player': 0,
            'percent_points_won_opponent': 0,
            'percent_first_serve_won_player': 0,
            'percent_first_serve_won_opponent': 0,
            'percent_second_serve_won_player': 0,
            'percent_second_serve_won_opponent': 0,
            'percent_first_serve_won_return_player': 0,
            'percent_first_serve_won_return_opponent': 0,
            'percent_second_serve_won_return_player': 0,
            'percent_second_serve_won_return_opponent': 0,
            'set_scores': []
        }
        while player_sets < self.num_sets_to_win and opponent_sets < self.num_sets_to_win:
            set_winner, p_games, o_games, set_points = self.simulate_set(starting_server)
            match_stats['player_games'] += p_games
            match_stats['opponent_games'] += o_games
            match_stats['points_played'] += set_points
            match_stats['set_scores'].append((p_games, o_games))
            if set_winner == 'Player':
                player_sets += 1
            else:
                opponent_sets += 1
            starting_server = 1 - starting_server
        # Derive stats from stat_outcomes
        match_stats['player_sets'] = player_sets
        match_stats['opponent_sets'] = opponent_sets
        match_stats['aces_player'] = self.stat_outcomes['player']['ace']['success']
        match_stats['aces_opponent'] = self.stat_outcomes['opponent']['ace']['success']
        match_stats['double_faults_player'] = self.stat_outcomes['player']['double_fault']['success']
        match_stats['double_faults_opponent'] = self.stat_outcomes['opponent']['double_fault']['success']
        match_stats['rallies_won_player'] = self.stat_outcomes['player']['rally_won']['success']
        match_stats['rallies_won_opponent'] = self.stat_outcomes['opponent']['rally_won']['success']
        match_stats['bp_faced_player'] = self.stat_outcomes['player']['bp_faced']['success']
        match_stats['bp_faced_opponent'] = self.stat_outcomes['opponent']['bp_faced']['success']
        match_stats['bp_saved_player'] = self.stat_outcomes['player']['bp_saved']['success']
        match_stats['bp_saved_opponent'] = self.stat_outcomes['opponent']['bp_saved']['success']
        match_stats['bp_won_player'] = self.stat_outcomes['player']['bp_won']['success']
        match_stats['bp_won_opponent'] = self.stat_outcomes['opponent']['bp_won']['success']
        serve_attempts_player = self.stat_outcomes['player']['serve_attempt']['success']
        serve_attempts_opponent = self.stat_outcomes['opponent']['serve_attempt']['success']
        first_serves_in_player = self.stat_outcomes['player']['first_serve_in']['success']
        first_serves_in_opponent = self.stat_outcomes['opponent']['first_serve_in']['success']
        match_stats['percent_first_serves_in_player'] = (
            first_serves_in_player / serve_attempts_player * 100 if serve_attempts_player > 0 else 0
        )
        match_stats['percent_first_serves_in_opponent'] = (
            first_serves_in_opponent / serve_attempts_opponent * 100 if serve_attempts_opponent > 0 else 0
        )
        match_stats['percent_bp_saved_player'] = (
            match_stats['bp_saved_player'] / match_stats['bp_faced_player'] * 100 if match_stats['bp_faced_player'] > 0 else 0
        )
        match_stats['percent_bp_saved_opponent'] = (
            match_stats['bp_saved_opponent'] / match_stats['bp_faced_opponent'] * 100 if match_stats['bp_faced_opponent'] > 0 else 0
        )
        match_stats['percent_bp_won_player'] = (
            match_stats['bp_won_player'] / match_stats['bp_faced_opponent'] * 100 if match_stats['bp_faced_opponent'] > 0 else 0
        )
        match_stats['percent_bp_won_opponent'] = (
            match_stats['bp_won_opponent'] / match_stats['bp_faced_player'] * 100 if match_stats['bp_faced_player'] > 0 else 0
        )
        # New stats
        serve_points_won_player = self.stat_outcomes['player']['serve_point_won']['success']
        serve_points_won_opponent = self.stat_outcomes['opponent']['serve_point_won']['success']
        return_points_won_player = self.stat_outcomes['player']['return_point_won']['success']
        return_points_won_opponent = self.stat_outcomes['opponent']['return_point_won']['success']
        total_points_won_player = self.stat_outcomes['player']['total_points_won']['success']
        total_points_won_opponent = self.stat_outcomes['opponent']['total_points_won']['success']
        first_serve_won_player = self.stat_outcomes['player']['first_serve_won']['success']
        first_serve_won_opponent = self.stat_outcomes['opponent']['first_serve_won']['success']
        second_serve_won_player = self.stat_outcomes['player']['second_serve_won']['success']
        second_serve_won_opponent = self.stat_outcomes['opponent']['second_serve_won']['success']
        first_serve_won_return_player = self.stat_outcomes['player']['first_serve_won_return']['success']
        first_serve_won_return_opponent = self.stat_outcomes['opponent']['first_serve_won_return']['success']
        second_serve_won_return_player = self.stat_outcomes['player']['second_serve_won_return']['success']
        second_serve_won_return_opponent = self.stat_outcomes['opponent']['second_serve_won_return']['success']
        first_serve_faced_player = self.stat_outcomes['player']['first_serve_faced']['success']
        first_serve_faced_opponent = self.stat_outcomes['opponent']['first_serve_faced']['success']
        second_serve_faced_player = self.stat_outcomes['player']['second_serve_faced']['success']
        second_serve_faced_opponent = self.stat_outcomes['opponent']['second_serve_faced']['success']
        match_stats['percent_serve_points_won_player'] = (
            serve_points_won_player / serve_attempts_player * 100 if serve_attempts_player > 0 else 0
        )
        match_stats['percent_serve_points_won_opponent'] = (
            serve_points_won_opponent / serve_attempts_opponent * 100 if serve_attempts_opponent > 0 else 0
        )
        match_stats['percent_return_points_won_player'] = (
            return_points_won_player / serve_attempts_opponent * 100 if serve_attempts_opponent > 0 else 0
        )
        match_stats['percent_return_points_won_opponent'] = (
            return_points_won_opponent / serve_attempts_player * 100 if serve_attempts_player > 0 else 0
        )
        match_stats['percent_points_won_player'] = (
            total_points_won_player / match_stats['points_played'] * 100 if match_stats['points_played'] > 0 else 0
        )
        match_stats['percent_points_won_opponent'] = (
            total_points_won_opponent / match_stats['points_played'] * 100 if match_stats['points_played'] > 0 else 0
        )
        match_stats['percent_first_serve_won_player'] = (
            first_serve_won_player / first_serves_in_player * 100 if first_serves_in_player > 0 else 0
        )
        match_stats['percent_first_serve_won_opponent'] = (
            first_serve_won_opponent / first_serves_in_opponent * 100 if first_serves_in_opponent > 0 else 0
        )
        second_serves_in_player = self.stat_outcomes['player']['2nd_in']['success']
        second_serves_in_opponent = self.stat_outcomes['opponent']['2nd_in']['success']
        match_stats['percent_second_serve_won_player'] = (
            second_serve_won_player / second_serves_in_player * 100 if second_serves_in_player > 0 else 0
        )
        match_stats['percent_second_serve_won_opponent'] = (
            second_serve_won_opponent / second_serves_in_opponent * 100 if second_serves_in_opponent > 0 else 0
        )
        match_stats['percent_first_serve_won_return_player'] = (
            first_serve_won_return_player / first_serve_faced_player * 100 if first_serve_faced_player > 0 else 0
        )
        match_stats['percent_first_serve_won_return_opponent'] = (
            first_serve_won_return_opponent / first_serve_faced_opponent * 100 if first_serve_faced_opponent > 0 else 0
        )
        match_stats['percent_second_serve_won_return_player'] = (
            second_serve_won_return_player / second_serve_faced_player * 100 if second_serve_faced_player > 0 else 0
        )
        match_stats['percent_second_serve_won_return_opponent'] = (
            second_serve_won_return_opponent / second_serve_faced_opponent * 100 if second_serve_faced_opponent > 0 else 0
        )
        # Calculate point dominance
        match_stats['point_dominance_player'] = (
            match_stats['percent_return_points_won_player'] / (100 - match_stats['percent_serve_points_won_player'])
            if (100 - match_stats['percent_serve_points_won_player']) > 0 else 0
        )
        match_stats['point_dominance_opponent'] = (
            match_stats['percent_return_points_won_opponent'] / (100 - match_stats['percent_serve_points_won_opponent'])
            if (100 - match_stats['percent_serve_points_won_opponent']) > 0 else 0
        )
        match_stats['winner'] = self.player_name if player_sets == self.num_sets_to_win else self.opponent_name
        match_stats['total_games'] = match_stats['player_games'] + match_stats['opponent_games']
        match_stats['rallies_played'] = match_stats['rallies_won_player'] + match_stats['rallies_won_opponent']
        match_stats['percent_rallies_won_player'] = match_stats['rallies_won_player'] / match_stats['rallies_played'] if match_stats['rallies_played'] > 0 else 0
        match_stats['percent_rallies_won_opponent'] = match_stats['rallies_won_opponent'] / match_stats['rallies_played'] if match_stats['rallies_played'] > 0 else 0
        return match_stats

    def simulate_many_matches_parallel(self, n_simulations=1000, n_workers=None, verbose=1):
        print(f"Simulating {self.player_name} vs. {self.opponent_name} match...")
        if n_workers is None:
            n_workers = multiprocessing.cpu_count() - 1
        results = Parallel(n_jobs=n_workers)(
            delayed(_simulate_match_worker)(self) for _ in range(n_simulations)
        )
        summary = defaultdict(list)
        set_score_counts = Counter()
        match_score_counts = Counter()
        stats_to_track = self.stats_to_track
        player_name = self.player_name
        opponent_name = self.opponent_name
        summary_detailed = {'match_details': []}
        aggregate_outcomes = {
            'player': {stat: {'success': 0, 'fail': 0} for stat in stats_to_track},
            'opponent': {stat: {'success': 0, 'fail': 0} for stat in stats_to_track}
        }
        for res in results:
            match_stats = res['match_stats']
            stat_outcomes = res['stat_outcomes']
            summary['winner'].append(match_stats['winner'])
            summary['player_sets'].append(match_stats['player_sets'])
            summary['opponent_sets'].append(match_stats['opponent_sets'])
            summary['total_games'].append(match_stats['total_games'])
            summary['player_games'].append(match_stats['player_games'])
            summary['opponent_games'].append(match_stats['opponent_games'])
            summary['points_played'].append(match_stats['points_played'])
            summary['aces_player'].append(match_stats['aces_player'])
            summary['aces_opponent'].append(match_stats['aces_opponent'])
            summary['double_faults_player'].append(match_stats['double_faults_player'])
            summary['double_faults_opponent'].append(match_stats['double_faults_opponent'])
            summary['rallies_won_player'].append(match_stats['rallies_won_player'])
            summary['rallies_won_opponent'].append(match_stats['rallies_won_opponent'])
            summary['percent_rallies_won_player'].append(100 * match_stats['percent_rallies_won_player'])
            summary['percent_rallies_won_opponent'].append(100 * match_stats['percent_rallies_won_opponent'])
            summary['bp_faced_player'].append(match_stats['bp_faced_player'])
            summary['bp_faced_opponent'].append(match_stats['bp_faced_opponent'])
            summary['bp_saved_player'].append(match_stats['bp_saved_player'])
            summary['bp_saved_opponent'].append(match_stats['bp_saved_opponent'])
            summary['bp_won_player'].append(match_stats['bp_won_player'])
            summary['bp_won_opponent'].append(match_stats['bp_won_opponent'])
            summary['percent_first_serves_in_player'].append(match_stats['percent_first_serves_in_player'])
            summary['percent_first_serves_in_opponent'].append(match_stats['percent_first_serves_in_opponent'])
            summary['percent_bp_saved_player'].append(match_stats['percent_bp_saved_player'])
            summary['percent_bp_saved_opponent'].append(match_stats['percent_bp_saved_opponent'])
            summary['percent_bp_won_player'].append(match_stats['percent_bp_won_player'])
            summary['percent_bp_won_opponent'].append(match_stats['percent_bp_won_opponent'])
            summary['percent_serve_points_won_player'].append(match_stats['percent_serve_points_won_player'])
            summary['percent_serve_points_won_opponent'].append(match_stats['percent_serve_points_won_opponent'])
            summary['percent_return_points_won_player'].append(match_stats['percent_return_points_won_player'])
            summary['percent_return_points_won_opponent'].append(match_stats['percent_return_points_won_opponent'])
            summary['point_dominance_player'].append(match_stats['point_dominance_player'])
            summary['point_dominance_opponent'].append(match_stats['point_dominance_opponent'])
            summary['percent_points_won_player'].append(match_stats['percent_points_won_player'])
            summary['percent_points_won_opponent'].append(match_stats['percent_points_won_opponent'])
            summary['percent_first_serve_won_player'].append(match_stats['percent_first_serve_won_player'])
            summary['percent_first_serve_won_opponent'].append(match_stats['percent_first_serve_won_opponent'])
            summary['percent_second_serve_won_player'].append(match_stats['percent_second_serve_won_player'])
            summary['percent_second_serve_won_opponent'].append(match_stats['percent_second_serve_won_opponent'])
            summary['percent_first_serve_won_return_player'].append(match_stats['percent_first_serve_won_return_player'])
            summary['percent_first_serve_won_return_opponent'].append(match_stats['percent_first_serve_won_return_opponent'])
            summary['percent_second_serve_won_return_player'].append(match_stats['percent_second_serve_won_return_player'])
            summary['percent_second_serve_won_return_opponent'].append(match_stats['percent_second_serve_won_return_opponent'])
            for set_score in match_stats['set_scores']:
                set_score_counts[set_score] += 1
            match_score_counts[tuple(match_stats['set_scores'])] += 1
            summary_detailed['match_details'].append(stat_outcomes.copy())
            summary['total_sets'].append(match_stats['player_sets'] + match_stats['opponent_sets'])
            for player_id in ['player', 'opponent']:
                for stat in stats_to_track:
                    counts = stat_outcomes[player_id][stat]
                    aggregate_outcomes[player_id][stat]['success'] += counts['success']
                    aggregate_outcomes[player_id][stat]['fail'] += counts['fail']
        # Calculate match score distribution as percentages
        total_matches = n_simulations
        match_score_distribution = {
            " ".join(f"{s[0]}-{s[1]}" for s in score): 100 * count / total_matches
            for score, count in match_score_counts.items()
        }
        # Find the most frequent match score
        if match_score_counts:
            most_frequent_match = max(match_score_counts.items(), key=lambda x: x[1])
            most_frequent_score = " ".join(f"{s[0]}-{s[1]}" for s in most_frequent_match[0])
            most_frequent_count = most_frequent_match[1]
        else:
            most_frequent_score = "N/A"
            most_frequent_count = 0
            
        n = n_simulations
        player_prob_win = np.mean(np.array(summary['winner']) == player_name)
        opponent_prob_win = np.mean(np.array(summary['winner']) == opponent_name)

        player_ci_lower, player_ci_upper = proportion_confint(int(player_prob_win * n), n, alpha=0.05, method='wilson')
        opponent_ci_lower, opponent_ci_upper = proportion_confint(int(opponent_prob_win * n), n, alpha=0.05, method='wilson')

        results_summary = {
            'player_win_probability': player_prob_win,
            'opponent_win_probability': opponent_prob_win,
            'player_win_upper_ci': player_ci_upper,
            'player_win_lower_ci': player_ci_lower,
            'opponent_win_upper_ci': opponent_ci_upper,
            'opponent_win_lower_ci': opponent_ci_lower,
            'average_sets_won': {
                player_name: np.mean(summary['player_sets']),
                opponent_name: np.mean(summary['opponent_sets']),
            },
            'average_games_won': {
                player_name: np.mean(summary['player_games']),
                opponent_name: np.mean(summary['opponent_games']),
            },
            'average_points_played': np.mean(summary['points_played']),
            'total_games_distribution': summary['total_games'],
            'match_score_counts': dict(match_score_counts),
            'set_score_distribution': dict(set_score_counts),
            'match_score_distribution': match_score_distribution,
            'most_frequent_match_score': most_frequent_score,
            'most_frequent_match_count': most_frequent_count,
            'raw_results': summary,
            'aggregate_outcomes': aggregate_outcomes,
            'detailed_match_distributions': summary_detailed,
            'total_sets_distribution': summary['total_sets']
        }

        if verbose:
            print(f"Summary of {n_simulations} simulated matches:")

            # Create table for match statistics
            table_data = [
                ['Probability of winning', results_summary['player_win_probability'], results_summary['opponent_win_probability']],
                ['lower_ci', player_ci_lower, opponent_ci_lower],
                ['high ci', player_ci_upper, opponent_ci_upper],
                ['Average sets won', results_summary['average_sets_won'][player_name], results_summary['average_sets_won'][opponent_name]],
                ["Average games won", results_summary['average_games_won'][player_name], results_summary['average_games_won'][opponent_name]],
                ["Aces", np.mean(summary['aces_player']), np.mean(summary['aces_opponent'])],
                ["Double Faults", np.mean(summary['double_faults_player']), np.mean(summary['double_faults_opponent'])],
                ["Break Points Faced", np.mean(summary['bp_faced_player']), np.mean(summary['bp_faced_opponent'])],
                ["Break Points Saved", np.mean(summary['bp_saved_player']), np.mean(summary['bp_saved_opponent'])],
                ["Break Points Won (Returning)", np.mean(summary['bp_won_player']), np.mean(summary['bp_won_opponent'])],
                ["Percentage First Serves In", np.mean(summary['percent_first_serves_in_player']), np.mean(summary['percent_first_serves_in_opponent'])],
                ["Percentage Break Points Saved", np.mean(summary['percent_bp_saved_player']), np.mean(summary['percent_bp_saved_opponent'])],
                ["Percentage Break Points Won", np.mean(summary['percent_bp_won_player']), np.mean(summary['percent_bp_won_opponent'])],
                ["Percentage Serve Points Won", np.mean(summary['percent_serve_points_won_player']), np.mean(summary['percent_serve_points_won_opponent'])],
                ["Percentage Return Points Won", np.mean(summary['percent_return_points_won_player']), np.mean(summary['percent_return_points_won_opponent'])],
                ["Point Dominance", np.mean(summary['point_dominance_player']), np.mean(summary['point_dominance_opponent'])],
                ["Percentage Points Won", np.mean(summary['percent_points_won_player']), np.mean(summary['percent_points_won_opponent'])],
                ["Percentage First Serve Won", np.mean(summary['percent_first_serve_won_player']), np.mean(summary['percent_first_serve_won_opponent'])],
                ["Percentage Second Serve Won", np.mean(summary['percent_second_serve_won_player']), np.mean(summary['percent_second_serve_won_opponent'])],
                ["Percentage First Serve Won (Return)", np.mean(summary['percent_first_serve_won_return_player']), np.mean(summary['percent_first_serve_won_return_opponent'])],
                ["Percentage Second Serve Won (Return)", np.mean(summary['percent_second_serve_won_return_player']), np.mean(summary['percent_second_serve_won_return_opponent'])],
            ]

            # Convert table_data to a Pandas DataFrame
            df = pd.DataFrame(table_data, columns=["Metric", player_name, opponent_name])

            # Round numerical columns to 3 decimal places
            df[player_name] = df[player_name].round(3)
            df[opponent_name] = df[opponent_name].round(3)

            #print("\nMatch Statistics Averages:")
            #print(df.to_string(index=False))  # Display the DataFrame without the index for a cleaner look
            print("\nMatch Statistics Averages:")
            try:
                # Apply custom styling for Jupyter Notebook or HTML-compatible environments
                styled_df = df.style.set_properties(**{
                    'background-color': '#D4F1F4',
                    'border-color': 'black',
                    'border-style': 'solid',
                    'border-width': '1px'
                }).set_properties(
                    subset=['Metric'],  # Target the "Metric" column
                    **{
                        'background-color': '#05445E',
                        'color': 'white',  # White text for readability on dark background
                        'font-weight': 'bold'
                    }
                ).set_table_styles([
                    {
                        'selector': 'th',
                        'props': [
                            ('background-color', '#05445E'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('text-align', 'center')
                        ]
                    },
                    {
                        'selector': 'tr:nth-child(even)',
                        'props': [('background-color', '#D4F1F4')]
                    },
                    {
                        'selector': 'tr:nth-child(odd)',
                        'props': [('background-color', '#D4F1F4')]
                    }
                ]).format({
                    player_name: "{:.3f}",
                    opponent_name: "{:.3f}"
                }).hide_index()  # Hide the index column
                display(styled_df)
            except (NameError, AttributeError):
                # Fallback for non-Jupyter environments (e.g., terminal)
                print(df.to_string(index=False))
                print("Note: Table styling is only visible in Jupyter Notebook or HTML-compatible environments.")

        
        
        return results_summary, styled_df

    def visualize_results(self, results):
        colors = ['#900020', '#05445E', '#297e95', '#4db8cc', '#8dd8de', '#D4F1F4']
        bar_colors = [colors[i % len(colors)] for i in range(6)]

        # Set score distribution as percentage
        set_score_counts = results['set_score_distribution']
        total_set_scores = sum(set_score_counts.values())
        set_score_labels = [f"{p}-{o}" for (p, o), _ in sorted(set_score_counts.items())]
        set_score_percentages = [100 * count / total_set_scores for _, count in sorted(set_score_counts.items())]
        bar_colors = [colors[i % len(colors)] for i in range(len(set_score_labels))]
        # Matches by number of sets as pie chart
        total_sets_counts = Counter(results['total_sets_distribution'])
        total_matches = len(results['total_sets_distribution'])
        set_totals = sorted(total_sets_counts.keys())
        set_percentages = [100 * count / total_matches for count in [total_sets_counts[total] for total in set_totals]]
        pie_colors = [colors[i % len(colors)] for i in range(len(set_totals))]
        prob_winning = np.array([results['player_win_probability'], results['opponent_win_probability']])
        player_labels = [self.player_name, self.opponent_name]
        
        fig1 = go.Figure()
        fig1.add_trace(
            go.Pie(
                labels=[f"{total}" for total in player_labels],
                values=prob_winning,
                text=[f"{val:.1f}%" for val in prob_winning],
                textposition='inside',
                marker=dict(colors=pie_colors),
                hovertemplate='%{label}<br>%{percent:.1f}%<extra></extra>'
            )
        )
        fig1.update_layout(
            title="Matches by Number of Sets",
            width=600,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig1.update_traces(textinfo='label+percent')
        st.plotly_chart(fig1)
        
       # Extract win probabilities and confidence intervals
        player_win_prob_pct = 100 * results['player_win_probability']
        opponent_win_prob_pct = 100 * results['opponent_win_probability']
        player_ci_lower_pct = player_win_prob_pct - 100 * results['player_win_lower_ci']
        player_ci_upper_pct = 100 * results['player_win_upper_ci'] - player_win_prob_pct
        opponent_ci_lower_pct = opponent_win_prob_pct - 100 * results['opponent_win_lower_ci']
        opponent_ci_upper_pct = 100 * results['opponent_win_upper_ci'] - opponent_win_prob_pct

        # Data for both players
        y_data = [player_win_prob_pct, opponent_win_prob_pct]
        x_data = [self.player_name, self.opponent_name]
        q1_data = [player_ci_lower_pct, opponent_ci_lower_pct]
        q3_data = [player_ci_upper_pct, opponent_ci_upper_pct]

        
        fig3 = go.Figure()
        fig3.add_trace(
            go.Pie(
                labels=[f"{total} Sets" for total in set_totals],
                values=set_percentages,
                text=[f"{val:.1f}%" for val in set_percentages],
                textposition='inside',
                marker=dict(colors=pie_colors),
                hovertemplate='%{label}<br>%{percent:.1f}%<extra></extra>'
            )
        )
        fig3.update_layout(
            title="Matches by Number of Sets",
            width=600,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig3.update_traces(textinfo='label+percent')
        st.plotly_chart(fig4)
        # Set distribution (stacked)
        opponent_sets = np.array(results['raw_results']['opponent_sets'])
        player_sets = np.array(results['raw_results']['player_sets'])
        stacked_v = np.stack([player_sets, opponent_sets], axis=1)
        tuple_counts = Counter(map(tuple, stacked_v))
        sorted_items = sorted(tuple_counts.items(), key=lambda x: (x[0][0], x[0][1]))
        labels = [str(k) for k, _ in sorted_items]
        counts = [v for _, v in sorted_items]
        fig4 = go.Figure()
        fig4.add_trace(
            go.Pie(
                labels=labels,
                values=counts,
                text=[f"{val:.1f}%" for val in counts],
                textposition='inside',
                marker=dict(colors=colors),
                hovertemplate='%{label}<br>%{percent:.1f}%<extra></extra>'
            )
        )
        fig4.update_layout(
            title="Set Distribution (Percentage)",
            width=800,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        fig4.update_traces(textinfo='label+percent')
        st.plotly_chart(fig4)
        # Set score distribution
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=set_score_labels,
                y=set_score_percentages,
                marker_color=bar_colors,
                text=[f"{val:.1f}%" for val in set_score_percentages],
                textposition="auto"
            )
        )
        fig2.update_layout(
            title="Set Score Distribution (Percentage)",
            xaxis_title="Set Scores",
            yaxis_title="Percentage (%)",
            width=800,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig2)
        # Total games histogram
        fig5 = go.Figure()
        fig5.add_trace(
            go.Histogram(
                x=results['total_games_distribution'],
                nbinsx=30,
                marker_color=colors[0],
                histnorm='percent',
                texttemplate="%{y:.1f}%",
                textposition="auto"
            )
        )
        fig5.update_layout(
            title="Distribution of Total Games Played",
            xaxis_title="Total Games Played",
            yaxis_title="Percentage of Matches (%)",
            width=600,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig5)
        # Cumulative distribution of total games
        data_sorted = np.sort(results['total_games_distribution'])
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        fig6 = go.Figure()
        fig6.add_trace(
            go.Scatter(
                x=data_sorted,
                y=cdf,
                mode='lines+markers',
                line=dict(color=colors[1]),
                marker=dict(size=5, color=colors[1])
            )
        )
        fig6.update_layout(
            title="Cumulative Probability Distribution of Total Games",
            xaxis_title="Total Games Played",
            yaxis_title="Cumulative Probability",
            width=600,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig6)
        # Stats KDE plots (existing + new)
        stats = [
            'aces', 'double_faults', 'percent_rallies_won', 'bp_faced', 'bp_saved', 'bp_won',
            'percent_first_serves_in', 'percent_bp_saved', 'percent_bp_won',
            'percent_serve_points_won', 'percent_return_points_won', 'point_dominance',
            'percent_points_won', 'percent_first_serve_won', 'percent_second_serve_won',
            'percent_first_serve_won_return', 'percent_second_serve_won_return'
        ]
        for stat in stats:
            fig7 = go.Figure()
            for player_type in ['player', 'opponent']:
                name = self.player_name if player_type == 'player' else self.opponent_name
                stat_key = f"{stat}_{player_type}"
                if stat_key in results['raw_results']:
                    data = np.array(results['raw_results'][stat_key])
                    if len(data) > 1 and np.std(data) > 0:
                        kde = gaussian_kde(data)
                        x_range = np.linspace(min(data), max(data), 200)
                        kde_values = kde(x_range)
                        total_area_approx = np.sum(kde_values) * (x_range[1] - x_range[0])  # Approximate integral
                        kde_percentages = (kde_values / total_area_approx) * 100
                        fig7.add_trace(
                            go.Scatter(
                                x=x_range,
                                y=kde_percentages,
                                mode='lines',
                                name=name,
                                line=dict(color=colors[0] if player_type == 'player' else colors[3])
                            )
                        )
                    else:
                        print(f"Not enough data or variance for KDE of {stat_key}")
                else:
                    print(f"Stat {stat_key} not found in results")
            fig7.update_layout(
                title=f"{self.player_name} vs {self.opponent_name} Distribution of {stat.replace('_', ' ').capitalize()}",
                xaxis_title=stat.replace('_', ' ').capitalize(),
                yaxis_title="Percentage occured (%)",
                width=600,
                height=400,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            st.plotly_chart(fig7)

# TournamentSimulator class (unchanged)
class TournamentSimulator:
    def __init__(self, players, df, num_sets_to_win=2, surface="Hard", rounds=None, seed_rankings=None):
        if not (len(players) & (len(players) - 1) == 0):
            raise ValueError(f"Number of players ({len(players)}) must be a power of 2")
        self.players = players
        self.df = df
        self.num_sets_to_win = num_sets_to_win
        self.surface = surface
        self.rounds = rounds or self._default_rounds(len(players))
        self.seed_rankings = seed_rankings or {p: i + 1 for i, p in enumerate(players)}
        self.results = []

    def _default_rounds(self, num_players):
        rounds = []
        while num_players > 1:
            rounds.append(f"Round of {num_players}")
            num_players //= 2
        rounds.append("Final")
        return rounds

    def create_bracket(self):
        n = len(self.players)
        sorted_players = sorted(self.players, key=lambda p: self.seed_rankings.get(p, float('inf')))
        bracket = [None] * n
        seed_positions = [0] + [n - i for i in range(1, n)]
        for i, pos in enumerate(seed_positions):
            bracket[pos] = sorted_players[i]
        return bracket

    def simulate_match(self, player1, player2, round_name):
        player1_sim = PlayerSimulator.from_match_data(player1, self.df, self.surface)
        player2_sim = PlayerSimulator.from_match_data(player2, self.df, self.surface)
        match = MatchSimulator(player1_sim, player2_sim, num_sets_to_win=self.num_sets_to_win, rounds=round_name)
        return match.simulate_match()

    def simulate_tournament(self):
        current_round = self.create_bracket()
        tournament_path = {player: [] for player in self.players}
        match_details = []
        round_idx = 0
        while len(current_round) > 1:
            next_round = []
            round_name = self.rounds[round_idx]
            for j in range(0, len(current_round), 2):
                player1 = current_round[j]
                player2 = current_round[j + 1]
                match_result = self.simulate_match(player1, player2, round_name)
                winner = match_result['winner']
                loser = player1 if winner == player2 else player2
                tournament_path[winner].append({
                    "opponent": loser,
                    "result": "win",
                    "match_data": match_result,
                    "round": round_name
                })
                tournament_path[loser].append({
                    "opponent": winner,
                    "result": "loss",
                    "match_data": match_result,
                    "round": round_name
                })
                match_details.append(match_result)
                next_round.append(winner)
            current_round = next_round
            round_idx += 1
        winner = current_round[0]
        tournament_path[winner].append({"result": "tournament_win", "round": "Final"})
        return winner, tournament_path, match_details

    def simulate_many_tournaments(self, n_simulations=1000):
        win_counter = Counter()
        round_progressions = defaultdict(Counter)
        all_tournament_paths = []
        all_match_details = []
        for _ in range(n_simulations):
            winner, tournament_path, match_details = self.simulate_tournament()
            win_counter[winner] += 1
            for player, rounds in tournament_path.items():
                for i, result in enumerate(rounds):
                    if isinstance(result, dict):
                        round_desc = result.get("result", "")
                        round_name = result.get("round", f"Round {i+1}")
                        round_progressions[player][f"{round_name}: {round_desc}"] += 1
            all_tournament_paths.append(tournament_path)
            all_match_details.append(match_details)
        tournament_summary = {
            "win_odds": {player: 100 * wins / n_simulations for player, wins in win_counter.items()},
            "progressions": {player: dict(rounds) for player, rounds in round_progressions.items()},
            "tournament_paths": all_tournament_paths,
            "match_details": all_match_details
        }
        return tournament_summary

    def visualize_results(self, results):
        round_wins = defaultdict(lambda: defaultdict(int))
        round_total = defaultdict(lambda: defaultdict(int))
        for tournament in results['match_details']:
            for match in tournament:
                round_ = match['round']
                p1 = match['player']
                p2 = match['opponent']
                winner = match['winner']
                round_total[p1][round_] += 1
                round_total[p2][round_] += 1
                if winner == p1:
                    round_wins[p1][round_] += 1
                elif winner == p2:
                    round_wins[p2][round_] += 1
        all_players = sorted(set(round_total.keys()) | set(round_wins.keys()))
        all_rounds = sorted(set(r for player in round_total.values() for r in player))
        data = []
        for player in all_players:
            row = []
            for round_ in all_rounds:
                wins = round_wins[player].get(round_, 0)
                total = round_total[player].get(round_, 0)
                pct = (wins / total * 100) if total > 0 else 0
                row.append(pct)
            data.append(row)
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=data,
                x=all_rounds,
                y=all_players,
                colorscale="RdYlGn",
                zmin=0,
                zmax=100,
                text=[[f"{val:.1f}%" for val in row] for row in data],
                texttemplate="%{text}",
                textfont=dict(size=10),
                colorbar=dict(title="Win %")
            )
        )
        win_odds = [results['win_odds'].get(p, 0) for p in all_players]
        fig.add_trace(
            go.Bar(
                x=[r + " (Win)" for r in all_rounds] + ["Tournament Win"],
                y=[0] * len(all_rounds) + win_odds,
                marker_color="#36A2EB",
                opacity=0.5,
                name="Tournament Win %"
            )
        )
        fig.update_layout(
            title="Player Win Percentage by Tournament Round",
            xaxis_title="Round",
            yaxis_title="Player",
            height=600,
            width=1000,
            showlegend=True
        )
        st.plotly_chart(fig)
        for metric in ['aces', 'double_faults', 'rallies', 'bp_faced', 'bp_saved', 'bp_won']:
            fig = go.Figure()
            for player_name in self.players:
                all_matches = [match for tournament in results['match_details'] for match in tournament]
                stat_player = [match[f'{metric}_player'] for match in all_matches if match['player'] == player_name]
                stat_opp = [match[f'{metric}_opponent'] for match in all_matches if match['opponent'] == player_name]
                tot_stat = stat_player + stat_opp
                if not tot_stat:
                    print(f"No data found for player: {player_name}")
                    continue
                fig.add_trace(
                    go.Histogram(
                        x=tot_stat,
                        name=player_name,
                        nbinsx=20,
                        opacity=0.5
                    )
                )
            fig.update_layout(
                title=f"{metric.capitalize()} Distribution per Match",
                xaxis_title=f"{metric.capitalize()} per Match",
                yaxis_title="Frequency",
                barmode="overlay",
                height=500,
                width=800,
                showlegend=True
            )
            fig.show()

def _simulate_match_worker(serialized_self):
    simulator = copy.deepcopy(serialized_self)
    simulator.reset_stat_outcomes()
    match_stats = simulator.simulate_match()
    return {
        'match_stats': match_stats,
        'stat_outcomes': simulator.stat_outcomes
    }
