import pickle

import pandas as pd
import plotly.express as px

MAX_NUM_MODELS = 30
DATA_SOURCES = ["acled", "wikipedia", "dbnomics", "fred", "yfinance"]
MARKET_SOURCES = [
    "manifold",
    "infer",
    "metaculus",
    "polymarket",
]


def abbreviate_unique(names, max_len=12):
    short_names = {}
    seen = set()
    for name in names:
        if len(name) <= max_len:
            short_names[name] = name
            seen.add(name)
        else:
            base = name[: max_len - 4]
            suffix = 1
            new_name = f"{base}~{suffix}"
            while new_name in seen:
                suffix += 1
                new_name = f"{base}~{suffix}"
            short_names[name] = new_name
            seen.add(new_name)
    return short_names


def compute_pairwise_win_fraction(
    battles,
    max_num_models=MAX_NUM_MODELS,
    value_col=None,
):
    # Times each model wins as Model A

    if value_col is not None:
        a_win_ptbl = pd.pivot_table(
            battles[battles["winner"] == "model_a"],
            index="model_a",
            columns="model_b",
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
        b_win_ptbl = pd.pivot_table(
            battles[battles["winner"] == "model_b"],
            index="model_a",
            columns="model_b",
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
        num_battles_ptbl = pd.pivot_table(
            battles,
            index="model_a",
            columns="model_b",
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )

    else:
        a_win_ptbl = pd.pivot_table(
            battles[battles["winner"] == "model_a"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        b_win_ptbl = pd.pivot_table(
            battles[battles["winner"] == "model_b"],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        num_battles_ptbl = pd.pivot_table(
            battles,
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (a_win_ptbl + b_win_ptbl.T) / (num_battles_ptbl + num_battles_ptbl.T)

    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    # prop_wins = prop_wins[:max_num_models]
    model_names = sorted(prop_wins.index.tolist())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    # prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    # prop_wins = prop_wins[:max_num_models]
    # model_names = list(prop_wins.keys())
    # row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col


def get_question_type_lookup(df):
    unique_questions = pd.concat(
        [
            df[df["source"].isin(DATA_SOURCES)].drop_duplicates(
                subset=[
                    "forecast_due_date",
                    "id",
                    "source",
                    "direction",
                    "resolution_date_a",
                    "resolution_date_b",
                ],
                ignore_index=True,
            ),
            df[df["source"].isin(MARKET_SOURCES)].drop_duplicates(
                subset=[
                    "forecast_due_date",
                    "id",
                    "source",
                    "direction",
                ],
                ignore_index=True,
            ),
        ]
    )
    unique_questions["type"] = unique_questions["source"].apply(
        lambda src: (
            "data" if src in DATA_SOURCES else ("market" if src in MARKET_SOURCES else "unknown")
        )
    )
    if (unique_questions["type"] == "unknown").any():
        raise ValueError("Should either be market or data (1).")

    question_type_lookup = {}
    question_n_lookup = {}
    for forecast_due_date in unique_questions["forecast_due_date"].unique():
        df_tmp = unique_questions[unique_questions["forecast_due_date"] == forecast_due_date]
        n_dataset_questions = (df_tmp["type"] == "data").sum()
        n_market_questions = (df_tmp["type"] == "market").sum()
        question_n_lookup[(forecast_due_date, "market")] = (n_dataset_questions, n_market_questions)
        question_type_lookup[(forecast_due_date, "market")] = 1.0 if n_market_questions else 0.0
        if n_market_questions and n_dataset_questions:
            question_type_lookup[(forecast_due_date, "data")] = (
                n_market_questions / n_dataset_questions
            )
        else:
            question_type_lookup[(forecast_due_date, "data")] = 1.0 if n_dataset_questions else 0.0
    return question_type_lookup


def get_type(df):
    df["type"] = df["source"].apply(
        lambda src: (
            "data" if src in DATA_SOURCES else ("market" if src in MARKET_SOURCES else "unknown")
        )
    )
    return df


def setup_equal_weights(df):
    question_type_lookup = get_question_type_lookup(df)
    df = get_type(df)

    df["per_row_weight"] = df.set_index(["forecast_due_date", "type"]).index.map(
        question_type_lookup
    )
    return df

def compute_pairwise_equal_weight_win_fraction(df, max_num_models=MAX_NUM_MODELS):
    df = setup_equal_weights(df)
    return compute_pairwise_win_fraction(df, value_col="per_row_weight", max_num_models=MAX_NUM_MODELS)


def compute_pairwise_equal_weight_scaled_win_fraction(df, max_num_models=MAX_NUM_MODELS):
    df = setup_equal_weights(df)
    df["score_diff"] = abs(df["score_a"] - df["score_b"])
    df["scale"] = df["per_row_weight"] * (1 + df["score_diff"])
    return compute_pairwise_win_fraction(df, value_col="scale", max_num_models=MAX_NUM_MODELS)

def visualize_pairwise_win_fraction(row_beats_col, title, max_num_models=MAX_NUM_MODELS):

    # Store original names
    full_index = sorted(row_beats_col.index.tolist())
    full_columns = sorted(row_beats_col.columns.tolist())

    # Generate shortened but unique names
    short_map = abbreviate_unique(full_columns)
    row_beats_col_short = row_beats_col.rename(index=short_map, columns=short_map)

    # Create customdata for full names
    custom_data = []
    for i in full_index:
        row = []
        for j in full_columns:
            row.append((i, j))
        custom_data.append(row)

    fig = px.imshow(
        row_beats_col_short,
        color_continuous_scale="RdBu",
        text_auto=".2f",
        title=title,
    )

    fig.update_traces(
        customdata=custom_data,
        hovertemplate=(
            "Model A: %{customdata[0]}<br>"
            "Model B: %{customdata[1]}<br>"
            "Wins   : %{z}<extra></extra>"
        ),
    )

    fig.update_layout(
        xaxis_title=" Model B: Loser",
        yaxis_title="Model A: Winner",
        xaxis_side="top",
        height=1700,
        width=1700,
        xaxis_tickangle=45,
        yaxis_tickfont=dict(size=10),
        xaxis_tickfont=dict(size=10),
        title_y=0.07,
        title_x=0.5,
    )

    return fig


def get_win_rate(df, title):
    row_beats_col = compute_pairwise_win_fraction(df, MAX_NUM_MODELS)
    fig = visualize_pairwise_win_fraction(row_beats_col, title=title)
    fig.write_html(f"{title}_winrate.html")


def get_equal_weight_win_rate(df, title):
    row_beats_col = compute_pairwise_equal_weight_win_fraction(df, MAX_NUM_MODELS)
    fig = visualize_pairwise_win_fraction(row_beats_col, title=title)
    fig.write_html(f"{title}_equal_weight_winrate.html")


def get_equal_weight_scaled_win_rate(df, title):
    row_beats_col = compute_pairwise_equal_weight_scaled_win_fraction(df, MAX_NUM_MODELS)
    fig = visualize_pairwise_win_fraction(row_beats_col, title=title)
    fig.write_html(f"{title}_equal_weight_scaled_winrate.html")


def read_pickle_and_get_winrate(filename, title, mask=None):
    with open(filename, "rb") as file:
        df = pickle.load(file)
    print(" * win/loss")
    get_win_rate(df, title)
    print(" * win/loss equal weight")
    get_equal_weight_win_rate(df, title)
    print(" * win/loss equal weight & scaled")
    get_equal_weight_scaled_win_rate(df, title)


if __name__ == "__main__":
    leaderboard_suffix = "human"
    elo_scores = {}
    for to_run in [
        "overall",
    ]:  # "data", "market", "market_resolved", "market_unresolved"]:
        print(f"Getting {to_run} win rate.")
        mask = None if to_run == "overall" else to_run
        elo_scores[to_run] = read_pickle_and_get_winrate(
            filename=f"df_{to_run}_{leaderboard_suffix}.pkl", title=to_run, mask=mask
        )
