"""Only a full identity and verified completion record may skip a run."""
import json
from collections import defaultdict

import numpy as np

from cloud2watt.evaluation.forecast import PROTOCOL, require_primary
from cloud2watt.run_state import is_complete, json_hash


def completed_formal_run(output_root, run_name, mode, seed, max_epochs, *, identity=None):
    if identity is None:
        return False
    if (identity.get("mode"), identity.get("seed"), identity.get("max_epochs")) != (
        mode, seed, max_epochs
    ):
        return False
    if (identity.get("max_train_samples") is not None
            or identity.get("max_validation_samples") is not None
            or identity.get("test_unlocked") is not False):
        return False
    return any(is_complete(path, identity)
               for path in output_root.glob(f"{run_name}-{mode}-s{seed}-*"))


def rank_completed_runs(paths, expected_seeds=(42, 123, 2026)):
    """Rank this development fold by the V2 metric; incomplete seed sets cannot win."""
    groups, population = defaultdict(list), None
    for path in paths:
        provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
        identity = provenance["identity"]
        if not is_complete(path, identity):
            raise ValueError("cannot rank incomplete or corrupt run")
        if identity["scope"] != "development":
            raise ValueError("smoke runs cannot enter candidate ranking")
        report = json.loads((path / "metrics.json").read_text(encoding="utf-8"))[
            "validation_development"]["none"]
        current = report["comparison"]
        if population is not None and population != current:
            raise ValueError("candidate comparison populations differ")
        population = current
        # The effective config includes seed; all other scientific/runtime identities
        # (including separate training/model config hashes) must agree across seeds.
        candidate = {k: v for k, v in identity.items()
                     if k not in ("seed", "effective_config_hash")}
        groups[json_hash(candidate)].append({"identity": identity,
                                            "score": require_primary(report), "path": str(path)})
    rows = []
    for candidate_id, runs in groups.items():
        seeds = [r["identity"]["seed"] for r in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate seed in candidate matrix")
        complete = set(seeds) == set(expected_seeds)
        rows.append({"candidate_id": candidate_id, "mode": runs[0]["identity"]["mode"],
                     "parameters": runs[0]["identity"]["parameters"],
                     "seeds": sorted(seeds), "selection_ready": complete,
                     "daylight_primary_mae": float(np.mean([r["score"] for r in runs])),
                     "runs": [{"seed": r["identity"]["seed"], "score": r["score"],
                               "path": r["path"]} for r in runs]})
    rows.sort(key=lambda r: (not r["selection_ready"], r["daylight_primary_mae"],
                             r["parameters"], r["candidate_id"]))
    return {"protocol": PROTOCOL, "scope": "single_development_fold",
            "selection_metric": "daylight_primary_mae", "candidates": rows,
            "note": "Not a multi-season or final-test victory; S3/S7/S8 remain required"}
