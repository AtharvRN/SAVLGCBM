import argparse
import json
import os

from evaluations.sparse_utils import sparsity_acc_test_savlg_cbm


DEFAULT_MEASURE_LEVEL = (5, 10, 15, 20, 25, 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, required=True)
    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.2,0.3,0.5")
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--n_iters", type=int, default=None)
    parser.add_argument("--max_glm_steps", type=int, default=150)
    parser.add_argument("--cbl_batch_size", type=int, default=None)
    parser.add_argument("--saga_batch_size", type=int, default=None)
    parser.add_argument("--disable_activation_cache", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    alpha_values = [float(x) for x in args.alphas.split(",") if x.strip()]
    results = []
    for alpha in alpha_values:
        accs = sparsity_acc_test_savlg_cbm(
            args.load_path,
            lam_max=args.lam,
            n_iters=args.n_iters,
            max_glm_steps=args.max_glm_steps,
            cbl_batch_size=args.cbl_batch_size,
            saga_batch_size=args.saga_batch_size,
            alpha_override=alpha,
            disable_activation_cache_override=args.disable_activation_cache,
        )
        row = {
            "alpha": alpha,
            "ACC@5": float(accs[0]),
            "AVGACC": float(sum(accs) / len(accs)),
        }
        for nec, acc in zip(DEFAULT_MEASURE_LEVEL, accs):
            row[f"ACC@{nec}"] = float(acc)
        results.append(row)

    payload = {
        "load_path": args.load_path,
        "lam": args.lam,
        "alphas": alpha_values,
        "results": results,
    }
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
