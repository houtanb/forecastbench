import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
    return compute_pairwise_win_fraction(
        df, value_col="per_row_weight", max_num_models=MAX_NUM_MODELS
    )


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


def use_elos(row_beats_col, df_leaderboard):
    score_lookup = df_leaderboard.set_index("org_model")["Score overall"].to_dict()
    for r_model in row_beats_col.index:
        for c_model in row_beats_col.columns:
            val = row_beats_col.loc[r_model, c_model]
            if not pd.isna(val):
                eta_m = score_lookup[r_model]
                eta_m_p = score_lookup[c_model]
                row_beats_col.loc[r_model, c_model] = val - (
                    1 / (1 + 10 ** ((eta_m_p - eta_m) / 400))
                )
    return row_beats_col


def plot_predicted_vs_expected_winrate(row_beats_col, df_leaderboard, leaderboard_suffix):
    pred = []
    actual = []
    hover_text = []
    score_lookup = df_leaderboard.set_index("org_model")["Score overall"].to_dict()
    for r_model in row_beats_col.index:
        for c_model in row_beats_col.columns:
            val = row_beats_col.loc[r_model, c_model]
            if not pd.isna(val):
                xi_m = score_lookup[r_model]
                xi_m_p = score_lookup[c_model]
                e_win_rate = expected_win_rate(xi_m_p - xi_m)
                pred.append(e_win_rate)
                actual.append(val)
                hover_text.append(
                    f"r_model: {r_model} ({round(xi_m, 2)})<br>c_model: {c_model} ({round(xi_m_p, 2)})"
                    f"<br>(x,y): ({round(e_win_rate, 2)},{round(val, 2)})"
                )

    def get_reg(pred, actual):
        slope, intercept = np.polyfit(pred, actual, 1)
        reg_x = np.linspace(min(pred), max(pred), 100)
        reg_y = slope * reg_x + intercept
        corr_coef = np.corrcoef(pred, actual)[0, 1]
        return reg_x, reg_y, round(corr_coef, 2)

    # winrates = [
    #     # (xi_diff, pred_win_rate)
    #     (3, 1-0.0474),
    #     (4, 1-0.0179),
    #     (5, 1-0.00669),
    #     (6, 1-0.00247),
    # ]

    pred_no_ex = []
    actual_no_ex = []
    hover_text_no_ex = []
    eps = 0.001
    for i, val in enumerate(pred):
        if val > 0.0 + eps and val < 1.0 - eps:
            pred_no_ex.append(val)
            actual_no_ex.append(actual[i])
            hover_text_no_ex.append(hover_text[i])

    reg_x, reg_y, corr_coef = get_reg(pred, actual)
    reg_x_no_ex, reg_y_no_ex, corr_coef_no_ex = get_reg(pred_no_ex, actual_no_ex)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=pred, y=actual, mode="markers", text=hover_text, hoverinfo="text", name="Orig")
    )
    fig.add_trace(go.Scatter(x=reg_x, y=reg_y, mode="lines", name=f"Regression Line ({corr_coef})"))
    fig.add_trace(
        go.Scatter(
            x=pred_no_ex,
            y=actual_no_ex,
            mode="markers",
            text=hover_text_no_ex,
            hoverinfo="text",
            name="No 0 or 1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=reg_x_no_ex,
            y=reg_y_no_ex,
            mode="lines",
            name=f"Regression Line No 0 or 1 ({corr_coef_no_ex})",
        )
    )
    fig.update_layout(
        title="Predicted vs. Actual Winrate",
        xaxis=dict(range=[0, 1], showgrid=True, gridcolor="lightgray"),
        yaxis=dict(range=[0, 1], showgrid=True, gridcolor="lightgray"),
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
    )
    fig.write_html(f"predicted_vs_actual_winrate_{leaderboard_suffix}.html")


def expected_win_rate(xi_diff):
    return 1 / (1 + 10 ** (xi_diff / 400))


def get_win_rate(df, title):
    row_beats_col = compute_pairwise_win_fraction(df, MAX_NUM_MODELS)
    fig = visualize_pairwise_win_fraction(row_beats_col, title=title)
    fig.write_html(f"{title}_winrate.html")


def get_win_rate_comparison(df, df_leaderboard, title, leaderboard_suffix):
    row_beats_col = compute_pairwise_win_fraction(df, MAX_NUM_MODELS)
    elo_row_beats_col = use_elos(row_beats_col.copy(), df_leaderboard)
    fig = visualize_pairwise_win_fraction(row_beats_col, title=title)
    fig.write_html(f"{title}_winrate_{leaderboard_suffix}.html")
    fig = visualize_pairwise_win_fraction(elo_row_beats_col, title=title)
    fig.write_html(f"{title}_winrate_minus_elo_{leaderboard_suffix}.html")
    plot_predicted_vs_expected_winrate(row_beats_col, df_leaderboard, leaderboard_suffix)


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
    # print(" * win/loss equal weight")
    # get_equal_weight_win_rate(df, title)
    # print(" * win/loss equal weight & scaled")
    # get_equal_weight_scaled_win_rate(df, title)


def read_pickle_and_csv_and_get_winrate(
    filename,
    title,
    leaderboard_suffix,
    mask,
):

    with open(filename, "rb") as file:
        df = pickle.load(file)

    with open(filename, "rb") as file:
        leaderboard_file = (
            "equal_weight_performance_scale/"
            + (f"{leaderboard_suffix}_" if leaderboard_suffix == "human" else "")
            + "leaderboard_overall.csv"
        )
        df_leaderboard = pd.read_csv(leaderboard_file)
        df_leaderboard["org_model"] = df_leaderboard["Organization"] + ";" + df_leaderboard["Model"]
        df_leaderboard = df_leaderboard[["org_model", "Score overall"]]

    print(" * win/loss")
    get_win_rate_comparison(
        df=df,
        df_leaderboard=df_leaderboard,
        title=title,
        leaderboard_suffix=leaderboard_suffix,
    )
    # print(" * win/loss equal weight")
    # get_equal_weight_win_rate(df, title)
    # print(" * win/loss equal weight & scaled")
    # get_equal_weight_scaled_win_rate(df, title)


if __name__ == "__main__":
    leaderboard_suffix = "human"
    elo_scores = {}
    elo_diffs = {}
    for to_run in [
        "overall",
    ]:  # "data", "market", "market_resolved", "market_unresolved"]:
        print(f"Getting {to_run} win rate.")
        mask = None if to_run == "overall" else to_run
        # elo_scores[to_run] = read_pickle_and_get_winrate(
        #     filename=f"df_{to_run}_{leaderboard_suffix}.pkl", title=to_run, mask=mask
        # )
        elo_diffs[to_run] = read_pickle_and_csv_and_get_winrate(
            filename=f"df_{to_run}_{leaderboard_suffix}.pkl",
            title=to_run,
            leaderboard_suffix=leaderboard_suffix,
            mask=mask,
        )
