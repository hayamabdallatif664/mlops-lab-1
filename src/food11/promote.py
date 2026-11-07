"""Quality gate: promote a registered model version to `champion` only if it is
clearly better than the current champion on the test split.

Usage: python src/food11/promote.py --version 3
"""

import argparse
import math
import os
import sys

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def test_accuracy(client, model_version, metric):
    metrics = client.get_run(model_version.run_id).data.metrics
    if metric not in metrics:
        sys.exit(f"run {model_version.run_id} (version {model_version.version}) has no metric {metric!r}")
    return metrics[metric]


def main():
    parser = argparse.ArgumentParser(description="Promote a model version if it passes the quality gate")
    parser.add_argument("--version", required=True, help="candidate model version number")
    parser.add_argument("--model-name", default="food11")
    parser.add_argument("--metric", default="test_accuracy")
    parser.add_argument(
        "--n",
        type=int,
        help="test split size (default: the champion run's test_images param)",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    candidate = client.get_model_version(args.model_name, args.version)
    acc_candidate = test_accuracy(client, candidate, args.metric)

    try:
        champion = client.get_model_version_by_alias(args.model_name, "champion")
    except MlflowException:
        champion = None

    if champion is None:
        # Nothing to compare against: the first model becomes champion
        client.set_registered_model_alias(args.model_name, "champion", candidate.version)
        client.set_model_version_tag(args.model_name, candidate.version, "gate_decision", "promoted (no champion)")
        print(f"No champion yet: version {candidate.version} promoted to champion")
        return

    if champion.version == candidate.version:
        print(f"Version {candidate.version} is already the champion, nothing to do")
        return

    acc_champion = test_accuracy(client, champion, args.metric)
    n = args.n or int(client.get_run(champion.run_id).data.params["test_images"])
    se = math.sqrt(acc_champion * (1 - acc_champion) / n)
    threshold = acc_champion + 2 * se

    if acc_candidate >= threshold:
        client.set_registered_model_alias(args.model_name, "champion", candidate.version)
        decision = "promoted"
    else:
        client.set_registered_model_alias(args.model_name, "challenger", candidate.version)
        decision = "challenger"

    tags = {
        "gate_decision": decision,
        "gate_candidate_accuracy": f"{acc_candidate:.4f}",
        "gate_champion_accuracy": f"{acc_champion:.4f}",
        "gate_champion_version": champion.version,
        "gate_threshold": f"{threshold:.4f}",
        "gate_se": f"{se:.4f}",
        "gate_n": str(n),
    }
    for key, value in tags.items():
        client.set_model_version_tag(args.model_name, candidate.version, key, value)

    print(f"candidate  v{candidate.version}: {args.metric} = {acc_candidate:.4f}")
    print(f"champion   v{champion.version}: {args.metric} = {acc_champion:.4f}  (n = {n})")
    print(f"se = sqrt(p(1-p)/n) = {se:.4f}  ->  threshold = champion + 2*se = {threshold:.4f}")
    if decision == "promoted":
        print(f"PROMOTED: champion -> v{candidate.version}")
    else:
        print(f"NOT promoted: v{candidate.version} gets alias 'challenger', champion stays v{champion.version}")


if __name__ == "__main__":
    main()
