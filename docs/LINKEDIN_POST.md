# LinkedIn post — GOAT Lab v1

Most Michael Jordan vs. LeBron James debates start with a conclusion and work backward.

I wanted to try the opposite.

Over the last few weeks, I built **GOAT Lab**, a reproducible basketball analytics
project that separates the evidence from the value judgments behind the debate.

The project compares peak, prime, longevity, regular-season value, playoff performance,
offense, defense, winning context, and cultural impact. I also built a playoff-series
expectation model with temporal cross-validation, froze the scoring hierarchy before
the final run, and sampled 250,000 different weight systems within fixed group caps.

The frozen production result was extremely close:

**LeBron James: 89.258985**  
**Michael Jordan: 89.143895**

LeBron finished first by only **0.115091 points** and won **60.1484%** of the sampled
weight systems.

The more interesting finding was not simply who finished first.

Across four approved scaling methods, the result split 2–2. Defense was the strongest
factor moving the result toward Jordan, while offense was the largest counterweight
moving it toward LeBron.

That means the honest conclusion is not that a model “proved” who the GOAT is. The
conclusion is that LeBron leads narrowly under the preregistered production model, but
the winner still depends on defensible choices about scaling and what greatness should
value.

I built the project with Python, pandas, scikit-learn, Plotly, Streamlit, Parquet, and a
release process that includes tests, frozen configurations, artifact hashes, a
machine-readable manifest, and explicit limitations.

Live dashboard: **[ADD STREAMLIT LINK]**  
GitHub repository: **[ADD GITHUB LINK]**

I would genuinely appreciate criticism of the methodology, especially the category
construction, scaling choices, and uncertainty treatment.

#SportsAnalytics #DataScience #MachineLearning #Python #Streamlit #BasketballAnalytics
