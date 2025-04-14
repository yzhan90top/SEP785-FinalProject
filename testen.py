
# Automatically batch-process the first N records in the shot log.
# For each batch of games, retrieve shotchartdetail (coordinates and zone info),
# automatically detect team_id, align based on clock time, and merge with shot_logs.
# Includes progress bar, rate limit handling, retry mechanism, and resume from partial results.

import pandas as pd
import requests
import time
import os
from tqdm import tqdm
from nba_api.stats.endpoints import shotchartdetail, boxscoretraditionalv2
from nba_api.stats.library.http import NBAStatsHTTP

# Simulate browser headers
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
})

def patched_request(*args, **kwargs):
    return session.request(*args, **kwargs)

NBAStatsHTTP._request = staticmethod(patched_request)

# Get team_id with retry
def get_team_id_for_player_in_game(player_id, game_id, retries=3):
    for attempt in range(retries):
        try:
            nba_game_id = f"00{game_id}"
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=nba_game_id)
            player_stats = boxscore.player_stats.get_data_frame()
            team_id = player_stats[player_stats['PLAYER_ID'] == player_id]['TEAM_ID'].values[0]
            return team_id
        except Exception as e:
            print(f"[Retry {attempt + 1}/{retries}] Failed to get team_id: player_id={player_id}, game_id={game_id}, Error: {e}")
            time.sleep(5)
    return None

def clock_to_seconds(clock_str):
    if isinstance(clock_str, str):
        minutes, seconds = map(int, clock_str.split(':'))
        return minutes * 60 + seconds
    return None

def fetch_shotchart_from_logs_df(logs, season="2014-15", time_tolerance=3):
    player_game_pairs = logs[['player_id', 'GAME_ID']].drop_duplicates()
    all_charts = []

    for _, row in tqdm(player_game_pairs.iterrows(), total=len(player_game_pairs), desc="Fetching shotchart data"):
        player_id = row['player_id']
        original_game_id = str(row['GAME_ID'])
        nba_game_id = f"00{original_game_id}"
        team_id = get_team_id_for_player_in_game(player_id, original_game_id)

        if team_id is None:
            continue

        try:
            chart = shotchartdetail.ShotChartDetail(
                player_id=player_id,
                game_id_nullable=nba_game_id,
                team_id=team_id,
                context_measure_simple='FGA',
                season_nullable=season,
                season_type_all_star='Regular Season'
            )
            chart_df = chart.get_data_frames()[0]
            chart_df['player_id'] = player_id
            chart_df['GAME_ID'] = original_game_id
            chart_df['SECONDS_LEFT'] = chart_df['MINUTES_REMAINING'] * 60 + chart_df['SECONDS_REMAINING']
            all_charts.append(chart_df)
            time.sleep(5)
        except Exception as e:
            print(f"Fetch failed: player_id={player_id}, game_id={nba_game_id}, Error: {e}")
            print("Waiting 20 seconds before continuing...")
            time.sleep(20)

    if not all_charts:
        return None

    charts_df = pd.concat(all_charts, ignore_index=True)
    charts_df['GAME_ID'] = charts_df['GAME_ID'].astype(str)
    logs['GAME_ID'] = logs['GAME_ID'].astype(str)

    merged = pd.merge(
        logs,
        charts_df,
        left_on=['GAME_ID', 'player_id', 'PERIOD'],
        right_on=['GAME_ID', 'PLAYER_ID', 'PERIOD'],
        suffixes=('_log', '_chart')
    )
    merged['time_diff'] = abs(merged['SECONDS_LEFT_log'] - merged['SECONDS_LEFT_chart'])
    merged_filtered = merged[merged['time_diff'] <= time_tolerance].copy()

    final = merged_filtered[[
        'GAME_ID', 'player_name', 'PERIOD', 'GAME_CLOCK', 'SHOT_CLOCK', 'DRIBBLES', 'TOUCH_TIME',
        'PTS_TYPE', 'SHOT_RESULT', 'CLOSEST_DEFENDER', 'CLOSE_DEF_DIST', 'FGM', 'PTS',
        'ACTION_TYPE', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y'
    ]].drop_duplicates()

    return final

def fetch_shotchart_batch_merge(log_csv_path, num_samples=20000, season="2014-15", batch_size=10, time_tolerance=3):
    logs = pd.read_csv(log_csv_path).head(num_samples).copy()
    logs['player_name'] = logs['player_name'].str.lower()
    logs['SECONDS_LEFT'] = logs['GAME_CLOCK'].apply(clock_to_seconds)
    logs['GAME_ID'] = logs['GAME_ID'].astype(str)

    player_game_pairs = logs[['player_id', 'GAME_ID']].drop_duplicates().reset_index(drop=True)
    processed_games_path = "processed_games.txt"
    partial_path = "merged_shot_data_partial.csv"

    # Load completed game_id + player_id combinations
    if os.path.exists(processed_games_path):
        with open(processed_games_path, 'r') as f:
            completed_pairs = set(line.strip() for line in f.readlines())
    else:
        completed_pairs = set()

    # Load existing partial file (for resume support)
    if os.path.exists(partial_path):
        print("Found existing partial file, continuing to append...")
        all_final = [pd.read_csv(partial_path)]
    else:
        all_final = []

    for i in tqdm(range(0, len(player_game_pairs), batch_size), desc="Processing games in batches"):
        batch_pairs = player_game_pairs.iloc[i:i + batch_size]
        batch_key_list = [f"{r['player_id']}_{r['GAME_ID']}" for _, r in batch_pairs.iterrows()]
        batch_pairs_filtered = batch_pairs[[key not in completed_pairs for key in batch_key_list]]

        if batch_pairs_filtered.empty:
            continue

        batch_logs = logs[logs.set_index(['player_id', 'GAME_ID']).index.isin(
            batch_pairs_filtered.set_index(['player_id', 'GAME_ID']).index)].copy()
        batch_result = fetch_shotchart_from_logs_df(batch_logs, season, time_tolerance)

        if batch_result is not None:
            all_final.append(batch_result)
            pd.concat(all_final).to_csv(partial_path, index=False)

        # Record completed game_id + player_id
        with open(processed_games_path, 'a') as f:
            for _, row in batch_pairs_filtered.iterrows():
                f.write(f"{row['player_id']}_{row['GAME_ID']}")

    if all_final:
        return pd.concat(all_final, ignore_index=True)
    return None

# Example usage
if __name__ == "__main__":
    df = fetch_shotchart_batch_merge(
        "../shot_logs.csv", # The shot_logs where you put
        num_samples=20000,
        season="2014-15",
        batch_size=10,
        time_tolerance=4
    )

    if df is not None:
        df.to_csv("merged_shot_data.csv", index=False)
        print(df.head())
    else:
        print("No shot data retrieved successfully. Possibly invalid game IDs.")
