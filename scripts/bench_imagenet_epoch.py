import argparse
import contextlib
import os
import time

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image, ImageFile, UnidentifiedImageError


def _now() -> float:
    return time.time()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--amp", action="store_true", help="Use AMP (fp16) for speed benchmarking.")
    p.add_argument("--data_parallel", action="store_true", help="Use torch.nn.DataParallel across all visible GPUs.")
    p.add_argument("--max_steps", type=int, default=0, help="If >0, stop after this many steps (for quick estimates).")
    args = p.parse_args()

    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[bench] device={device} n_gpus={n_gpus} amp={args.amp} data_parallel={args.data_parallel}", flush=True)

    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    min_bytes = int(os.environ.get("CBM_MIN_IMAGE_BYTES", "0") or "0")

    def _safe_loader(path: str) -> Image.Image:
        try:
            if min_bytes > 0:
                try:
                    if os.path.getsize(path) < min_bytes:
                        raise UnidentifiedImageError("file too small")
                except OSError:
                    raise UnidentifiedImageError("stat failed")
            with Image.open(path) as img:
                return img.convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Keep benchmark running even if a few files are corrupted.
            return Image.new("RGB", (224, 224), (0, 0, 0))

    ds = datasets.ImageFolder(args.train_root, transform=tfm, loader=_safe_loader)
    print(f"[bench] dataset_len={len(ds)} classes={len(ds.classes)} root={args.train_root}", flush=True)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    model = models.resnet50(weights=None)
    model.to(device)
    if args.data_parallel and device == "cuda" and n_gpus > 1:
        model = nn.DataParallel(model)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=args.amp and device == "cuda")

    global_step = 0
    for epoch in range(args.epochs):
        t0 = _now()
        seen = 0
        first_batch_time_s = None
        for step, (images, target) in enumerate(loader, start=1):
            if first_batch_time_s is None:
                first_batch_time_s = _now() - t0
                print(f"[bench] time_to_first_batch_s={first_batch_time_s:.2f}", flush=True)
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            amp_ctx = (
                autocast(dtype=torch.float16, enabled=scaler.is_enabled())
                if device == "cuda"
                else contextlib.nullcontext()
            )
            with amp_ctx:
                logits = model(images)
                loss = loss_fn(logits, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bsz = int(images.shape[0])
            seen += bsz
            global_step += 1

            if step == 1 or step % 10 == 0 or step % 50 == 0:
                dt = max(_now() - t0, 1e-6)
                ips = seen / dt
                # ETA for this epoch (rough): based on steps.
                total_steps = len(loader)
                eta_s = (total_steps - step) * (dt / step)
                print(
                    f"[bench] epoch={epoch+1} step={step}/{total_steps} "
                    f"loss={float(loss.item()):.4f} ips={ips:.1f} ETA~{eta_s/60.0:.1f}m",
                    flush=True,
                )

            if args.max_steps > 0 and global_step >= args.max_steps:
                print("[bench] stopping early due to --max_steps", flush=True)
                break

        dt = max(_now() - t0, 1e-6)
        ips = seen / dt
        print(f"[bench] epoch_done={epoch+1} time_s={dt:.1f} images={seen} ips={ips:.1f}", flush=True)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
