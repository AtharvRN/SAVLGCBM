import os
from argparse import ArgumentParser

import pandas as pd

from evaluations.sparse_utils import (
    sparsity_acc_test,
    sparsity_acc_test_lf_cbm,
    sparsity_acc_test_salf_cbm,
    sparsity_acc_test_savlg_cbm,
)
from methods.common import load_run_info

parser = ArgumentParser()
parser.add_argument("--load_path", type=str)
parser.add_argument("--lam", type=float, default=0.1)
parser.add_argument("--filter", type=float, default=0)
parser.add_argument("--annotation_dir", type=str, default=None)
parser.add_argument("--result_file", type=str, default=None)
parser.add_argument("--lf-cbm", action="store_true")
parser.add_argument("--n_iters", type=int, default=None)
parser.add_argument("--max_glm_steps", type=int, default=None)

args = parser.parse_args()
run_info = load_run_info(args.load_path)
model_name = "lf_cbm" if args.lf_cbm else run_info.get("model_name", "vlg_cbm")
if model_name == "lf_cbm":
    accs = sparsity_acc_test_lf_cbm(
        args.load_path,
        lam_max=args.lam,
        n_iters=args.n_iters,
        max_glm_steps=args.max_glm_steps if args.max_glm_steps is not None else 150,
    )
elif model_name == "vlg_cbm":
    accs = sparsity_acc_test(
        args.load_path,
        lam_max=args.lam,
        bot_filter=args.filter,
        anno=args.annotation_dir,
        n_iters=args.n_iters,
        max_glm_steps=args.max_glm_steps if args.max_glm_steps is not None else 150,
    )
elif model_name == "salf_cbm":
    accs = sparsity_acc_test_salf_cbm(
        args.load_path,
        lam_max=args.lam,
        n_iters=args.n_iters,
        max_glm_steps=args.max_glm_steps if args.max_glm_steps is not None else 150,
    )
elif model_name == "savlg_cbm":
    accs = sparsity_acc_test_savlg_cbm(
        args.load_path,
        lam_max=args.lam,
        n_iters=args.n_iters,
        max_glm_steps=args.max_glm_steps if args.max_glm_steps is not None else 150,
    )
else:
    raise NotImplementedError(
        f"Sparse evaluation for model_name={model_name} is not implemented yet."
    )
if args.result_file:
    if os.path.exists(args.result_file):
        df = pd.read_csv(args.result_file)
    else:
        df = pd.DataFrame(columns=["ACC@5", "AVGACC"])
    row = pd.Series({"ACC@5": accs[0], "AVGACC": sum(accs) / len(accs)})
    df.loc[len(df.index)] = row
    df.to_csv(args.result_file, index=False)
