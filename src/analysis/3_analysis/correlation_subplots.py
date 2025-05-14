"""Generate correlation subplots for 2024-07-21 survey round."""

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_subplot_figure(human_or_llm, date):
    """Create subplot and bar chart."""
    if human_or_llm == "human_":
        subplot_title = f"{date} Human question set"
        add_to_filename = "human"
    elif human_or_llm == "":
        subplot_title = f"{date} LLM question set"
        add_to_filename = "llm"
    else:
        raise ValueError(human_or_llm)

    cols = ["Organization", "Model", "Ranking"]
    df7 = pd.read_csv(f"{date}_{human_or_llm}leaderboard_7_days_or_less.csv")[cols].rename(
        columns={"Ranking": "Rank7"}
    )
    df7_30 = pd.read_csv(f"{date}_{human_or_llm}leaderboard_30_days_or_less.csv")[cols].rename(
        columns={"Ranking": "Rank730"}
    )
    df7_30_90 = pd.read_csv(f"{date}_{human_or_llm}leaderboard_90_days_or_less.csv")[cols].rename(
        columns={"Ranking": "Rank73090"}
    )
    df7_30_90_180 = pd.read_csv(f"{date}_{human_or_llm}leaderboard_180_days_or_less.csv")[
        cols
    ].rename(columns={"Ranking": "Rank73090180"})

    def get_n(df):
        col = next(c for c in df.columns if "Dataset Score" in c)
        n_match = re.search(r"\(N=([\d,]+)\)", col)
        return int(n_match.group(1).replace(",", "")) if n_match else None

    n_vals = {}
    n_vals["Rank7"] = get_n(pd.read_csv(f"{date}_{human_or_llm}leaderboard_7_days_or_less.csv"))
    n_vals["Rank730"] = get_n(pd.read_csv(f"{date}_{human_or_llm}leaderboard_30_days_or_less.csv"))
    n_vals["Rank73090"] = get_n(
        pd.read_csv(f"{date}_{human_or_llm}leaderboard_90_days_or_less.csv")
    )
    n_vals["Rank73090180"] = get_n(
        pd.read_csv(f"{date}_{human_or_llm}leaderboard_180_days_or_less.csv")
    )

    df = (
        df7.merge(df7_30, on=["Organization", "Model"])
        .merge(df7_30_90, on=["Organization", "Model"])
        .merge(df7_30_90_180, on=["Organization", "Model"])
    )

    def make_bar_plot(df):
        diff7 = (df["Rank7"] - df["Rank73090180"]).abs()
        diff30 = (df["Rank730"] - df["Rank73090180"]).abs()
        diff90 = (df["Rank73090"] - df["Rank73090180"]).abs()
        med_df = pd.DataFrame(
            {
                "horizon": ["7-day", "30-day", "90-day"],
                "median_change": [diff7.median(), diff30.median(), diff90.median()],
            }
        )
        fig = px.bar(
            med_df,
            x="horizon",
            y="median_change",
            labels={"horizon": "Horizon", "median_change": "Median change in rank (abs. val.)"},
            template="plotly_white",
            text="median_change",
        )
        fig.update_traces(
            marker_color="steelblue", texttemplate="%{text:.0f}", textposition="outside"
        )
        fig.update_layout(
            title=f"Median Change in Rank by Horizon: {subplot_title}",
            yaxis=dict(range=[0, med_df["median_change"].max() * 1.1]),
        )
        fig.write_html(
            f"rank_median_change_{date}_{add_to_filename}.html",
            include_plotlyjs="cdn",
            config={"displayModeBar": False},
        )

    make_bar_plot(df=df)

    pairs = [
        (1, 1, "Rank7", "Rank730"),
        (2, 1, "Rank7", "Rank73090"),
        (3, 1, "Rank7", "Rank73090180"),
        (1, 2, None, None),
        (2, 2, "Rank730", "Rank73090"),
        (3, 2, "Rank730", "Rank73090180"),
        (1, 3, None, None),
        (2, 3, None, None),
        (3, 3, "Rank73090", "Rank73090180"),
    ]
    pairs.sort(key=lambda x: x[0])

    def top_k_retention(xcol, ycol, k=25):
        df_top_k = df[df[xcol] <= k]
        retained = (df_top_k[ycol] <= k).sum()
        return retained / k * 100

    titles = [
        ("7-day", "Rank7"),
        ("30-day", "Rank730"),
        ("90-day", "Rank73090"),
        ("180-day", "Rank73090180"),
    ]
    subplot_titles = []
    for row, col, x, y in pairs:
        subplot_text = ""
        if row == 1:
            titles_key = titles[:-1][col - 1]
            N = n_vals[titles_key[1]]
            subplot_text = titles_key[0] + f" (N={N:,})<br> "
        if x is not None and y is not None:
            corr = df[x].corr(df[y], method="spearman")
            k = 25
            ret_rate = top_k_retention(xcol=x, ycol=y, k=k)
            subplot_text += f"(r={corr:.2f}, Top {k} ret. {int(ret_rate)}%)"
        subplot_titles.append(subplot_text)

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )

    for r, c, xcol, ycol in pairs:
        if xcol is None or ycol is None:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    marker={"opacity": 0},
                    showlegend=False,
                ),
                row=r,
                col=c,
            )
            idx = (r - 1) * 3 + c
            xdom = fig.layout[f"xaxis{idx}"].domain
            ydom = fig.layout[f"yaxis{idx}"].domain
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=xdom[0],
                x1=xdom[1],
                y0=ydom[0],
                y1=ydom[1],
                fillcolor="white",
                line_width=0,
                layer="below",
            )
            fig.update_xaxes(
                row=r,
                col=c,
                showgrid=False,
                zeroline=False,
                showline=False,
                showticklabels=False,
            )
            fig.update_yaxes(
                row=r,
                col=c,
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="white",
                ticks="",
                tickfont=dict(color="white"),
            )
            continue

        # scatter + lines
        m, b = np.polyfit(df[xcol], df[ycol], 1)
        xs = np.array([df[xcol].min(), df[xcol].max()])
        ys = m * xs + b

        fig.add_trace(
            go.Scatter(
                x=df[xcol],
                y=df[ycol],
                mode="markers",
                marker=dict(color="MediumSeaGreen", symbol="circle"),
                hovertext=df["Organization"] + "<br>" + df["Model"],
                name="",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=xs,
                mode="lines",
                line=dict(color="SteelBlue", dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color="Tomato", dash="solid"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(
            row=r,
            col=c,
            showgrid=False,
            zeroline=False,
        )
        fig.update_yaxes(
            row=r,
            col=c,
            showgrid=False,
            zeroline=False,
        )

    for i, (txt, n_key) in enumerate(titles[1:]):
        N = n_vals[n_key]
        fig.update_yaxes(title_text=f"{txt} (N={N:,})", row=i + 1, col=1)

    fig.update_layout(
        width=1600,
        height=1000,
        plot_bgcolor="GhostWhite",
        title={
            "text": subplot_title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20},
        },
    )

    max_rank = max(
        df["Rank7"].max(),
        df["Rank730"].max(),
        df["Rank73090"].max(),
        df["Rank73090180"].max(),
    )
    for r in range(1, 4):
        for c in range(1, 4):
            fig.update_xaxes(row=r, col=c, range=[0, max_rank + 1])
            fig.update_yaxes(row=r, col=c, range=[0, max_rank + 1])

    fig.write_html(
        f"rank_correlations_{date}_{add_to_filename}.html",
        include_plotlyjs="cdn",
        config={"displayModeBar": False},
    )


if __name__ == "__main__":
    date = "2024-07-21"
    make_subplot_figure(human_or_llm="human_", date=date)
    make_subplot_figure(human_or_llm="", date=date)
