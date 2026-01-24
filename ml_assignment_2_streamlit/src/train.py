"""Train all required models, evaluate, and persist artifacts."""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_dataset
from .models import get_models
from .metrics import evaluate_binary


def main(out_dir: str = 'model', random_state: int = 42):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataset(as_frame=True)
    # Save a copy to data folder for user reference
    data_path = out.parent / 'data' / 'breast_cancer_wdbc.csv'
    df.to_csv(data_path, index=False)

    feature_cols = [c for c in df.columns if c not in ('target', 'target_label')]
    X = df[feature_cols]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    models = get_models(random_state=random_state)
    rows = []
    artifacts = {}

    for name, model in models.items():
        if model is None:
            continue
        model.fit(X_train, y_train)
        m, cm, report = evaluate_binary(model, X_test, y_test)
        rows.append({'ML Model Name': name, **m})

        model_file = out / (name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') + '.joblib')
        joblib.dump(model, model_file)
        artifacts[name] = {
            'model_file': str(model_file),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'metrics': m
        }

    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[['ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    metrics_df.to_csv(out/'model_comparison_metrics.csv', index=False)

    with open(out/'artifacts.json', 'w', encoding='utf-8') as f:
        json.dump(artifacts, f, indent=2)

    # Also store the train/test split for reproducibility
    X_test.assign(target=y_test).to_csv(out/'holdout_test_set.csv', index=False)

    return metrics_df


if __name__ == '__main__':
    main()
