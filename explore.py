"""
Position Scarcity Model - Cycle 6, Day 6 project
Extends the Day 3 Monte Carlo swap method across multiple Argentina players,
grouped by position, to see where losing a key player hurts most.

Limitation: this only measures goal+assist output, the same limitation as
Day 3. Defenders and defensive midfielders will likely show almost no
simulated impact here, not because they don't matter, but because this
method can't see tackles, interceptions, or buildup play, only direct goal
involvement. This is a real limitation of the method, not a finding about
those players' value.

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

import pandas as pd
import numpy as np

matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)

TEAM = "Argentina"
N_SIMS = 10000
rng = np.random.default_rng(42)

arg_matches = matches[(matches["home_team"] == TEAM) | (matches["away_team"] == TEAM)].copy()
arg_matches["team_goals"] = arg_matches.apply(
    lambda r: r["home_score"] if r["home_team"] == TEAM else r["away_score"], axis=1
)
arg_matches["opp_goals"] = arg_matches.apply(
    lambda r: r["away_score"] if r["home_team"] == TEAM else r["home_score"], axis=1
)

actual_points = 0
for _, row in arg_matches.iterrows():
    if row["team_goals"] > row["opp_goals"]:
        actual_points += 3
    elif row["team_goals"] == row["opp_goals"]:
        actual_points += 1
print(f"\nArgentina's actual Copa America 2024 points: {actual_points}")

# ---------- position bucket per player ----------
def bucket_position(pos):
    if pd.isna(pos):
        return None
    pos = str(pos)
    if "Goalkeeper" in pos:
        return "Goalkeeper"
    if "Back" in pos:
        return "Defender"
    if "Midfield" in pos:
        return "Midfielder"
    if "Forward" in pos or "Wing" in pos or "Striker" in pos:
        return "Forward"
    return None

arg_events = events[events["team"] == TEAM].dropna(subset=["position"])
player_position = (
    arg_events.groupby("player")["position"]
    .agg(lambda x: x.mode().iloc[0])
    .apply(bucket_position)
)
player_position = player_position[player_position.notna() & (player_position != "Goalkeeper")]

# ---------- goal + assist contribution per player per match ----------
goals = events[(events["type"] == "Shot") & (events["shot_outcome"] == "Goal") & (events["team"] == TEAM)]
goals_per_match = goals.groupby(["match_id", "player"]).size().rename("goals")

assists = events[(events["pass_goal_assist"] == True) & (events["team"] == TEAM)]
assists_per_match = assists.groupby(["match_id", "player"]).size().rename("assists")

contrib = pd.concat([goals_per_match, assists_per_match], axis=1).fillna(0)
contrib["g_plus_a"] = contrib["goals"] + contrib["assists"]
contrib = contrib.reset_index()
contrib_lookup = contrib.set_index(["match_id", "player"])["g_plus_a"]

total_contrib = contrib.groupby("player")["g_plus_a"].sum().sort_values(ascending=False)
key_players = [p for p in total_contrib.index if p in player_position.index][:8]

print(f"\nTop {len(key_players)} contributors selected for simulation:")
for p in key_players:
    print(f"  {p} ({player_position[p]}): {total_contrib[p]:.0f} total G+A")

# ---------- run the Day 3 swap simulation for each key player ----------
results = []
for player in key_players:
    pos = player_position[player]
    teammates_same_pos = player_position[(player_position == pos) & (player_position.index != player)].index.tolist()

    replacement_pool_values = np.array([
        contrib_lookup.get((mid, teammate), 0)
        for mid in arg_matches["match_id"]
        for teammate in teammates_same_pos
    ])
    if len(replacement_pool_values) == 0:
        replacement_pool_values = np.array([0])

    player_by_match = {
        mid: contrib_lookup.get((mid, player), 0) for mid in arg_matches["match_id"]
    }

    sim_points_totals = np.zeros(N_SIMS)
    for _, row in arg_matches.iterrows():
        mid = row["match_id"]
        team_goals = row["team_goals"]
        opp_goals = row["opp_goals"]
        player_val = player_by_match[mid]

        sampled = rng.choice(replacement_pool_values, size=N_SIMS, replace=True)
        sim_team_goals = np.maximum(team_goals - player_val + sampled, 0)
        sim_points_totals += np.where(
            sim_team_goals > opp_goals, 3,
            np.where(sim_team_goals == opp_goals, 1, 0)
        )

    results.append({
        "player": player,
        "position": pos,
        "replacement_depth": len(teammates_same_pos),
        "sim_mean_points": sim_points_totals.mean(),
        "point_drop": actual_points - sim_points_totals.mean(),
    })

results_df = pd.DataFrame(results).sort_values("point_drop", ascending=False)
results_df.to_csv("position_scarcity_results.csv", index=False)

print("\nPoint drop if each player's contribution was replaced (higher = harder to replace):")
print(results_df[["player", "position", "replacement_depth", "point_drop"]].to_string(index=False))

# ---------- visualization ----------
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#0d1b2a")
ax.set_facecolor("#0d1b2a")

pos_colors = {"Forward": "#f4a300", "Midfielder": "#4a90d9", "Defender": "#6fcf97"}
colors = [pos_colors.get(p, "#999999") for p in results_df["position"]]

# use recognizable display names, avoid duplicate surnames (two Martínez) and
# legal family names nobody recognizes (Messi's is Cuccittini)
name_overrides = {
    "Lionel Andrés Messi Cuccittini": "Messi",
    "Lautaro Javier Martínez": "Lautaro Martínez",
    "Lisandro Martínez": "Lisandro Martínez",
    "Alexis Mac Allister": "Mac Allister",
    "Giovani Lo Celso": "Lo Celso",
}
def display_name(full_name):
    return name_overrides.get(full_name, full_name.split()[-1])

labels = [display_name(p) for p in results_df["player"]]

bars = ax.barh(labels, results_df["point_drop"], color=colors)
ax.invert_yaxis()

# give the x-axis headroom on both sides so negative bars (point drop < 0,
# meaning the sim says the team does better without that player) aren't
# clipped at zero and rendered invisible
min_val = min(results_df["point_drop"].min(), 0)
max_val = results_df["point_drop"].max()
span = max_val - min_val
ax.set_xlim(min_val - span * 0.15, max_val + span * 0.15)
ax.axvline(0, color="#5a6a7a", linewidth=1)

for bar, depth in zip(bars, results_df["replacement_depth"]):
    width = bar.get_width()
    offset = 6 if width >= 0 else -6
    ha = "left" if width >= 0 else "right"
    ax.annotate(f"depth: {depth}", (width, bar.get_y() + bar.get_height() / 2),
                textcoords="offset points", xytext=(offset, 0), va="center", ha=ha,
                color="white", fontsize=9)

ax.set_xlabel("Simulated point drop if replaced (actual points minus simulated mean)\nnegative = model says team does better without them (a model limitation, see README)",
              color="white", fontsize=10.5)
fig.suptitle("Where Does Losing a Key Player Hurt Most?", color="white", fontsize=15, y=0.97)
ax.set_title("Copa America 2024, point drop by position when swapping each player's G+A output",
             color="#a8b8c8", fontsize=10, pad=10)
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_color("#3a4a5a")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in pos_colors.values()]
ax.legend(handles, pos_colors.keys(), facecolor="#0d1b2a", labelcolor="white", edgecolor="#3a4a5a", loc="lower right")

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("position_scarcity_model.png", dpi=150, facecolor=fig.get_facecolor())
print("\nSaved chart: position_scarcity_model.png")