import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from evaluations.sparse_utils import MAX_GLM_STEP, _extract_concepts, run_nec_eval
from model.cbm import load_eval_cbm

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str)
parser.add_argument("--lam", type=float, default=0.1)
parser.add_argument("--annotation_dir", type=str, default=None)
parser.add_argument("--result_file", type=str, default=None)
parser.add_argument("--lf-cbm", action="store_true")
parser.add_argument("--n_iters", type=int, default=None)
parser.add_argument("--max_glm_steps", type=int, default=None)
parser.add_argument("--cbl_batch_size", type=int, default=None)
parser.add_argument("--saga_batch_size", type=int, default=None)
parser.add_argument("--disable_activation_cache", action="store_true")
parser.add_argument("--eval_num_workers", type=int, default=None)
parser.add_argument("--max_images", type=int, default=None)

args = parser.parse_args()
max_glm_steps = args.max_glm_steps if args.max_glm_steps is not None else MAX_GLM_STEP

model = load_eval_cbm(
    args.load_path,
    annotation_dir=args.annotation_dir,
    lf_cbm=args.lf_cbm,
    cbl_batch_size=args.cbl_batch_size,
    saga_batch_size=args.saga_batch_size,
    disable_activation_cache=args.disable_activation_cache,
    eval_num_workers=args.eval_num_workers,
)

# Override saga_batch_size for non-savlg models (savlg handles it in from_pretrained)
if args.saga_batch_size is not None and getattr(model, "model_name", None) != "savlg_cbm":
    model.args.saga_batch_size = args.saga_batch_size

# Extract concept activations for each split
train_loader, val_loader, test_loader = model.get_data_loaders()
device = model.args.device


def _truncate_loader(loader):
    if loader is None or args.max_images is None:
        return loader
    max_images = int(args.max_images)
    if max_images <= 0 or len(loader.dataset) <= max_images:
        return loader
    subset = Subset(loader.dataset, list(range(max_images)))
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=False,
    )


train_loader = _truncate_loader(train_loader)
val_loader = _truncate_loader(val_loader)
test_loader = _truncate_loader(test_loader)

train_c, train_l = _extract_concepts(model.get_concept_activations, train_loader, device)
val_c = val_l = None
if val_loader is not None:
    val_c, val_l = _extract_concepts(model.get_concept_activations, val_loader, device)
test_c, test_l = _extract_concepts(model.get_concept_activations, test_loader, device)

accs = run_nec_eval(
    args.load_path, model.concepts, model.classes, model.args,
    train_c, train_l, val_c, val_l, test_c, test_l,
    lam_max=args.lam, n_iters=args.n_iters, max_glm_steps=max_glm_steps,
)

if args.result_file:
    if os.path.exists(args.result_file):
        df = pd.read_csv(args.result_file)
    else:
        df = pd.DataFrame(columns=["ACC@5", "AVGACC"])
    df.loc[len(df.index)] = {"ACC@5": accs[0], "AVGACC": sum(accs) / len(accs)}
    df.to_csv(args.result_file, index=False)
