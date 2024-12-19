import math
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

pd.options.display.float_format = "{:.2f}".format

MAX_NUM_MODELS = 30


def compute_pairwise_win_fraction(battles, max_num_models=30):
    # Times each model wins as Model A
    a_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting times each model wins as Model B
    b_win_ptbl = pd.pivot_table(
        battles[battles["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    # Table counting number of A-B pairs
    num_battles_ptbl = pd.pivot_table(
        battles, index="model_a", columns="model_b", aggfunc="size", fill_value=0
    )

    # Computing the proportion of wins for each model as A and as B
    # against all other models
    row_beats_col_freq = (a_win_ptbl + b_win_ptbl.T) / (num_battles_ptbl + num_battles_ptbl.T)

    # Arrange ordering according to proprition of wins
    prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
    prop_wins = prop_wins[:max_num_models]
    model_names = list(prop_wins.keys())
    row_beats_col = row_beats_col_freq.loc[model_names, model_names]
    return row_beats_col


def visualize_pairwise_win_fraction(battles, title, max_num_models=30):
    fig = px.imshow(row_beats_col, color_continuous_scale="RdBu", text_auto=".2f", title=title)
    fig.update_layout(
        xaxis_title=" Model B: Loser",
        yaxis_title="Model A: Winner",
        xaxis_side="top",
        height=900,
        width=900,
        title_y=0.07,
        title_x=0.5,
    )
    fig.update_traces(
        hovertemplate="Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>"
    )

    return fig


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None):
    from sklearn.linear_model import LogisticRegression

    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def compare_elos(row_beats_col, elo_mle_ratings):
    score_lookup = elo_mle_ratings.to_dict()
    for r_model in row_beats_col.index:
        for c_model in row_beats_col.columns:
            val = row_beats_col.loc[r_model, c_model]
            if not pd.isna(val):
                xi_m = score_lookup[r_model]
                xi_m_p = score_lookup[c_model]
                row_beats_col.loc[r_model, c_model] = val - (1 / (1 + np.exp(xi_m_p - xi_m)))
    return row_beats_col


def plot_predicted_vs_expected_winrate(row_beats_col, elo_mle_ratings):
    pred = []
    actual = []
    hover_text = []
    score_lookup = elo_mle_ratings.to_dict()
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
    fig.write_html("predicted_vs_actual_winrate.html")


def expected_win_rate(xi_diff):
    return 1 / (1 + 10 ** (xi_diff / 400))


def plot_expected_winrate():

    xi_diff = np.arange(0, 21, 1)
    winrate = expected_win_rate(xi_diff)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xi_diff, y=winrate, mode="lines", line=dict(width=2)))
    fig.update_layout(
        title="Expected Winrate vs. xi_diff",
        xaxis_title="xi_diff",
        yaxis_title="Expected Winrate",
        template="plotly_white",
        font=dict(size=12),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
    )
    fig.write_html("expected_winrate.html")


if __name__ == "__main__":
    filename = "chatbot_arena_file_name.json"
    if not os.path.exists(filename):
        url = "https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json"
        response = requests.get(url)

        with open(filename, "wb") as file:
            file.write(response.content)

    with open(filename, "r") as file:
        battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])

    # we use anony battles only for leaderboard
    battles = battles[battles["anony"] == True]  # noqa: E712

    # we de-duplicate top 0.1% redudant prompts
    # see https://lmsys.org/blog/2024-05-17-category-hard/#note-enhancing-quality-through-de-duplication
    print("Before dedup: ", len(battles))
    battles = battles[battles["dedup_tag"].apply(lambda x: x.get("sampled", False))]
    print("After dedup: ", len(battles))

    battles_no_ties = battles[~battles["winner"].str.contains("tie")]

    row_beats_col = compute_pairwise_win_fraction(battles, MAX_NUM_MODELS)
    # fig = visualize_pairwise_win_fraction(
    #     row_beats_col, title="Fraction of Model A Wins for All Non-tied A vs. B Battles"
    # )
    # fig.write_html("chatbot_arena_winrate.html")

    elo_mle_ratings = compute_mle_elo(battles)
    plot_predicted_vs_expected_winrate(row_beats_col, elo_mle_ratings)

    # row_beats_col = compare_elos(row_beats_col, elo_mle_ratings)
    # fig = visualize_pairwise_win_fraction(row_beats_col, title="Fraction wins - expected wins")
    # fig.write_html("chatbot_arena_winrate_minus_expected_winrate.html")

    # plot_expected_winrate()
