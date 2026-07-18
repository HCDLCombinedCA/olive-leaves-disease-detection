#!/usr/bin/env bash
# Set up CSIRO's leaf_segmenter_public (Mask R-CNN pretrained on synthetic
# Arabidopsis). This is LEGACY code: Matterport Mask R-CNN on TensorFlow 1.x /
# Keras, which does NOT run on Python 3.12 + modern TF. It needs an isolated
# Python 3.7 environment. This script clones the repos and prints the steps.
set -e
cd "$(dirname "$0")"

echo "== Cloning CSIRO leaf_segmenter_public =="
if [ ! -d upstream/leaf_segmenter_public ]; then
  git clone https://bitbucket.csiro.au/scm/ag3d/leaf_segmenter_public.git \
    upstream/leaf_segmenter_public || {
      echo "Bitbucket clone failed. Mirror/alternative:"
      echo "  https://github.com/DanielCWard/Deep-Leaf-Segmentation-Using-Synthetic-Data"
    }
fi

echo "== Cloning Matterport Mask R-CNN (dependency) =="
if [ ! -d upstream/Mask_RCNN ]; then
  git clone https://github.com/matterport/Mask_RCNN.git upstream/Mask_RCNN
fi

cat <<'EOF'

Next steps (needs conda or pyenv for Python 3.7 — this code is TF1-era):

  conda create -n csiro-leaf python=3.7 -y
  conda activate csiro-leaf
  pip install "tensorflow==1.15" "keras==2.2.5" numpy scipy Pillow \
              cython scikit-image imgaug opencv-python "h5py<3"
  pip install -e upstream/Mask_RCNN

Then download the pretrained leaf model weights (.h5) linked from the CSIRO
Synthetic Arabidopsis Dataset page and run the provided script in
upstream/leaf_segmenter_public (see its README for the exact entrypoint):

  https://research.csiro.au/robotics/databases/synthetic-arabidopsis-dataset/

EOF
echo "Clone step done. See this folder's README.md for context and caveats."
