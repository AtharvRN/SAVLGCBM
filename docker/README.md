# CBM Image (Build Once, Run Jobs Fast)

This folder contains a Dockerfile that bakes the `cbm` conda environment into an image.

Why:
- Creating/solving conda environments at job runtime is slow and sometimes unreliable.
- A prebuilt image makes k8s jobs reproducible and starts faster.

## Build (Apple Silicon Mac)

You must build a Linux amd64 image for Nautilus:

```bash
cd /path/to/SAVLGCBM
docker buildx create --use

# Example tag (use your own registry/repo)
IMG=ghcr.io/<YOUR_GH_USERNAME>/savlgcbm-cbm:cu121

docker buildx build --platform linux/amd64 \
  -f docker/Dockerfile.cbm \
  -t "$IMG" \
  --push .
```

## Use In A Job

Set the job container image to `$IMG` and remove runtime env creation.

Because the image exports `PATH=/opt/conda/envs/cbm/bin:$PATH`, you can run:

```bash
python -u train_cbm.py ...
```

