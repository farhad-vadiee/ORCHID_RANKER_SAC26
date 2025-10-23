# Orchid Ranker (SAC'26 Artifact)

This anonymised artifact accompanies the SAC'26 submission on
agentic adaptive recommendation. Orchid Ranker is an adaptive educational
recommender toolkit that pairs
rich dataset preprocessing pipelines with a modular slate orchestration
engine, learner simulators, and a plug-and-play recommender class you can
drop straight into your product (much like `surprise`’s algorithms).
The toolkit grew out of experiments with the EdNet and OULAD datasets and now bundles:

- preprocessing utilities (`orchid_ranker.preprocessing`) that transform
  raw learner interaction logs into feature-rich CSV bundles;
- agent modules (`orchid_ranker.agents`) implementing student
  simulators, recommender policies, and the multi-user orchestration
  loop;
- ready-to-use baseline recommenders plus an `OrchidRecommender`
  high-level API that feels similar to `surprise`; and
- utilities for running offline experiments similar to those used in the
  associated LAK'26 studies.

## Installation

```bash
pip install .
```

or, once published, simply:

```
pip install orchid-ranker
```

## Quick start

### As a plug-and-play recommender

```python
import pandas as pd
from orchid_ranker import OrchidRecommender, Recommendation

# 1. Interaction log (implicit labels or ratings)
interactions = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 3, 3, 3],
    "item_id": [10, 12, 10, 11, 12, 13, 14],
    "label":   [1,  1,  1,  0,  1,  0,  1],
})

# Optional: item-side features (align with sorted item ids)
item_side = pd.DataFrame({
    "item_id": [10, 11, 12, 13, 14],
    "difficulty": [0.35, 0.80, 0.55, 0.25, 0.60],
    "popularity": [120, 45, 85, 30, 65],
})
item_features = (
    item_side[["difficulty", "popularity"]]
    .to_numpy(dtype="float32")
)

# 2. Instantiate a Surprise-style recommender (choose any strategy)
strategies = [
    ("linucb", {"alpha": 1.5, "item_features": item_features}),
    ("als", {"epochs": 5}),
    ("popularity", {}),
    ("random", {}),
]

for strategy, kwargs in strategies:
    rec = OrchidRecommender(strategy=strategy, **{k: v for k, v in kwargs.items() if k not in {"item_features"}})
    rec.fit(
        interactions,
        rating_col="label",
        item_features=kwargs.get("item_features"),
    )
    print(f"{strategy.title()} recommendations:", rec.recommend(user_id=1, top_k=5))

# Predict a specific score if needed
als_rec = OrchidRecommender(strategy="als", epochs=5).fit(interactions, rating_col="label")
print("ALS predicted relevance:", als_rec.predict(user_id=1, item_id=10))
```

### Running adaptive vs. baseline experiments

```python
from orchid_ranker.preprocessing import preprocess_ednet
from orchid_ranker.experiments import RankingExperiment

# 1) Preprocess your raw EdNet dump
preprocess_ednet(
    base_path="/path/to/raw/u-files",
    content_path="/path/to/content",
    output_path="./data/ednet-processed",
)

# 2) Run a quick comparison with the experiment driver
runner = RankingExperiment("configs/ednet.yaml", dataset="ednet", cohort_size=16)
summary = runner.run_many(["adaptive", "fixed", "linucb", "als"], dp_enabled=False)
print(summary)
```

See the `experiments/` directory for end-to-end experiment scripts and the
`runs/` folder for generated reports.

## Dataset format at a glance

See `data/README.md` for licensing notes and preprocessing instructions.

You can plug in **any** dataset as long as you provide five CSV files and
a short YAML schema:

- `train.csv`, `val.csv`, `test.csv` each with at least `u`, `i`, `label`
  (optionally `timestamp`, `correct`, `accept`, etc.).
- `side_information_users.csv` describing per-learner features.
- `side_information_items.csv` describing per-item features.
- `configs/<your-dataset>.yaml` declaring which columns are categorical
  vs numeric and where the CSVs live.

Minimal YAML example:

```yaml
run:
  dataset: my_dataset

datasets:
  my_dataset:
    paths:
      base_dir: data/my-dataset-processed
      train: train.csv
      val: val.csv
      test: test.csv
      side_information_users: side_information_users.csv
      side_information_items: side_information_items.csv
    interactions:
      timestamp: true
    users:
      categorical: [cohort, gender]
      numeric: [mean_accuracy, activity_span_days]
    items:
      categorical: [module, topic]
      numeric: [difficulty, recent_clicks_4w]
```

As long as these files exist, `orchid_ranker.data.DatasetLoader` handles
all encoding automatically.

## Visualising your data

The `orchid_ranker.visualization` module provides lightweight helpers:

```python
from orchid_ranker.visualization import (
    plot_user_activity,
    plot_item_difficulty,
    plot_learning_curve,
)

plot_user_activity(interactions_df, top_n=25)
plot_item_difficulty(items_df)
plot_learning_curve(round_summary_df, metric="mean_accuracy")
```

Each function returns a Matplotlib axes so you can further customise the
plot before saving it.


You can also toggle differential privacy quickly via presets:

```python
from orchid_ranker.dp import get_dp_config
from orchid_ranker.experiments import RankingExperiment

runner = RankingExperiment("configs/ednet.yaml", dataset="ednet")
summary = runner.run_many(["adaptive", "fixed"], dp_params=get_dp_config("eps_05"))
```


### Built-in baseline modes

`RankingExperiment` understands the following non-adaptive policies out of the box:

| Mode        | Description                                  |
|-------------|----------------------------------------------|
| `fixed`     | Two-tower recommender without online updates |
| `popularity`| Mean acceptance per item                     |
| `random`    | Uniform slate sampling                       |
| `als`       | Matrix-factorization baseline (trained once) |
| `implicit_als` | Weighted implicit ALS via the `implicit` package |
| `implicit_bpr` | Bayesian Personalized Ranking optimiser (`implicit`) |
| `neural_mf` | Shallow neural matrix factorisation with MLP head |
| `user_knn`  | User-based collaborative filtering           |
| `linucb`    | Linear contextual UCB over item features     |

Run them via `runner.run_many([...])` and everyone will report the same summary metrics.


To replicate the full battery of adaptive vs fixed comparisons on EdNet and OULAD run:

```bash
python experiments-sac/run_all.py
```

Results are written under the `runs/` folder (summary CSVs plus per-round metrics).
