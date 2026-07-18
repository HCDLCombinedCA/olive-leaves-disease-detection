#!/usr/bin/env bash
# Clone and set up PointsToWood (harryjfowen/PointsToWood) — deep-learning
# leaf-wood separation for TLS/LiDAR point clouds, with pretrained weights.
set -e
cd "$(dirname "$0")"

if [ ! -d upstream/PointsToWood ]; then
  echo "== Cloning PointsToWood =="
  git clone https://github.com/harryjfowen/PointsToWood.git upstream/PointsToWood
else
  echo "PointsToWood already cloned; pulling latest."
  git -C upstream/PointsToWood pull --ff-only || true
fi

cat <<'EOF'

Done. To run (uses PyTorch + torch-geometric — make a dedicated venv):

  cd upstream/PointsToWood
  python3 -m venv .venv && source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt        # see repo; installs torch + torch-geometric

  # predict on your cloud (0 = leaf, 1 = wood, plus wood-probability column):
  python predict.py --point-cloud /path/to/your_plot.ply

Notes:
  * Input: high-resolution TLS/LiDAR point cloud (.ply). Put yours in ../../data/.
  * The repo ships pretrained weights trained on diverse European forests, so no
    training needed to start. Check its README for the exact weights flag.
  * torch-geometric wheels must match your torch/CUDA version — follow the
    PyTorch Geometric install guide if pip resolution fails.
EOF
