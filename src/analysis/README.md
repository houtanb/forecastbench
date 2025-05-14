# ForecastBench

The following analysis investigates the stability of the dataset questions on the ForecastBench leaderboard.

## Survey leaderboards

We generated leaderboards for each survey round, scoring dataset questions by average Brier score. We use the average Brier score across all questions that have resolved, regardless of the forecast horizon.

To generate the leaderboards, we modified the data as follows:

* We limited all rounds here to include only those questions that resolved at the 7- and 30-day horizons.
* We dropped all prompts that provide freeze values as these do not impact forecasts on dataset questions.
* We dropped all combination questions.

Note that there are two 2024-07-21 leaderboards, one for the LLM question set, containing 250 standard dataset questions per forecast horizon and one for the human question set, containing 50 standard dataset questions per forecast horizon. This is the only round for which we ran a survey of superforecasters and the general public.

The leaderboards can be found here:

* [`1_leaderboards_by_due_date/2024-07-21_human_leaderboard_overall.html`](1_leaderboards_by_due_date/2024-07-21_human_leaderboard_overall.html)
* [`1_leaderboards_by_due_date/2024-07-21_leaderboard_overall.html`](1_leaderboards_by_due_date/2024-07-21_leaderboard_overall.html)
* [`1_leaderboards_by_due_date/2025-03-02_leaderboard_overall.html`](1_leaderboards_by_due_date/2025-03-02_leaderboard_overall.html)
* [`1_leaderboards_by_due_date/2025-03-16_leaderboard_overall.html`](1_leaderboards_by_due_date/2025-03-16_leaderboard_overall.html)
* [`1_leaderboards_by_due_date/2025-03-30_leaderboard_overall.html`](1_leaderboards_by_due_date/2025-03-30_leaderboard_overall.html)

The column `Pairwise p-value comparing to No. 1 (bootstrapped)` compares every model to the best-performing model, showing the p-value calculated by bootstrapping the differences in dataset Brier score between each model and the best forecaster (the group with rank 1) under the null hypothesis that there's no difference. 

You can see in `2024-07-21_human_leaderboard_overall.html` that there's a significant difference between every model's performance and superforecaster performance. Further, `2025-03-30_leaderboard_overall.html` shows that there is _not_ a significant difference in forecasting ability on dataset questions between DeepSeek-V3 (scratchpad), Claude-3-7-Sonnet-20250219 (scratchpad), and O3-Mini-2025-01-31 (zero shot).

In `2024-07-21_human_leaderboard_overall.html` you can also see that Claude-3-5-Sonnet-20240620 (with various prompts) performs at least as well as the general public on dataset questions, as does Mistral-Large-Latest (scratchpad). To investigate further, we changed the comparison group to the general public and included all available resolution dates (7, 30, 90, 180 days), to generate: [`1_leaderboards_by_due_date/2024-07-21_human_leaderboard_overall_public_comparison.html`](1_leaderboards_by_due_date/2024-07-21_human_leaderboard_overall_public_comparison.html). Here we see that we cannot reject the null that there's no difference between the general public and those ranked 16 and better.

## Unified leaderboard

From the above survey rounds, we selected all dataset questions that have resolved at the 7-day and 30-day horizons (i.e. we ignored the 90-day horizons and 180-day horizons from the 2024-07-21 survey round). We then generated a single leaderboard using the 2-way fixed effects model, decomposing performance into forecaster ability and question difficulty, then calculated the difficulty-adjusted Brier score.

You can view the unified leaderboard here: [`2_unified_leaderboard/2fe_human_leaderboard_overall.html`](2_unified_leaderboard/2fe_human_leaderboard_overall.html).

Superforecasters still appear at the top of the leaderboard; however, the 95% confidence intervals are quite wide and we cannot differentiate performance between Superforecasters and the other models ranked in the top 11. We believe the confidence intervals are wide because of lack of overlap in the datasets and the need to obtain more superforecaster observations. It's important to note that in calculating the CIs, we assumed normality, which likely is violated given our data, and hence the CIs presented are likely tighter than they would be if we bootstrapped them.

NB: we have not yet settled on a scoring mechanism for the leaderboard. Other scoring rules under consideration at the moment are: the Brier Skill Score, Elo, and the Peer score.

## Analysis

The goal of the analysis is to see how rankings (by Brier score) correlate across forecast horizons.

We limit our analysis to the oldest survey round run on July 21, 2024 since it has the most forecast horizons that have resolved: 7, 30, 90, and 180 days.

We create two figures to show the analysis, one for the human question set and one for the LLM question set:

* [`3_analysis/rank_correlations_2024-07-21_human.html`](3_analysis/rank_correlations_2024-07-21_human.html)
* [`3_analysis/rank_correlations_2024-07-21_llm.html`](3_analysis/rank_correlations_2024-07-21_llm.html)

In what follows we discuss the human question set but similar patterns are found for both question sets.

The columns of the figure are associated with the x-axes of each subplot. They are the model rankings at the 7-, 30-, and 90-day forecast horizons. The rows of the figure are associated with the y-axes of each subplot, and are the rankings at the 30-, 90-, and 180-day forecast horizons.

`N` is the total number of dataset questions that have resolved at the given forecast horizon, `r` is Spearman's rank correlation coefficient, and `Top 25 ret.` is the percentage of models ranked in the Top 25 at forecast horizon X (denoted by the column header) that are _still_ in the Top 25 in forecast horizon Y (denoted by the row header). So, in row 3, column 2 of `rank_correlations_2024-07-21_human.html`, 84% of models in the Top 25 at the 30-day forecast horizon were still in the Top 25 at the 180-day forecast horizon.

The subplots show that 80% of models in the top 25 at day 7 stay in the top 25 by day 180. So we can already begin to draw conclusions (with caveats) of model performance 7 days after forecasts have been submitted. To draw stronger conclusions, we should wait until the 30-day mark, as we see the high correlation between the 30-day rankings and both the 90-day and 180-day rankings (`r=0.98` and `r=0.96`). There appears to be little movement beyond the 90-day mark.

Further, we create two bar plots showing the median of the absolute value of the change in ranking from the 7-, 30-, and 90-day forecast horizons to the 180-day forecast horizon.

* [`3_analysis/rank_median_change_2024-07-21_human.html`](3_analysis/rank_median_change_2024-07-21_human.html)
* [`3_analysis/rank_median_change_2024-07-21_llm.html`](3_analysis/rank_median_change_2024-07-21_llm.html)

Again, you can see the same pattern with the rankings becoming more stable as time passes, with models typically moving by ±4 spots in the rankings between 30 days and 180 days after the forecasts were submitted. After 90 days, models typically move by ±1 spot.

## Current leaderboards
For reference, you can see the current leaderboards (updated nightly) on the ForecastBench website:

* [https://www.forecastbench.org/leaderboards/human_leaderboard_overall.html](https://www.forecastbench.org/leaderboards/human_leaderboard_overall.html)
* [https://www.forecastbench.org/leaderboards/leaderboard_overall.html](https://www.forecastbench.org/leaderboards/leaderboard_overall.html)

Click on the column entitled "Dataset Score" to sort the models by the dataset score.
