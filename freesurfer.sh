#!/bin/bash
set -euo pipefail

FS_SUBJECT_DIR=${1:?Need FreeSurfer subject dir (contains mri/ surf/)}
OUTDIR=${2:?Need output dir}

mkdir -p "$OUTDIR"

SUBJECT_ID=$(basename "$FS_SUBJECT_DIR")
export SUBJECTS_DIR=$(dirname "$FS_SUBJECT_DIR")

mri_surf2vol \
  --so "$FS_SUBJECT_DIR/surf/lh.white" "$FS_SUBJECT_DIR/surf/lh.curv" \
  --subject "$SUBJECT_ID" --identity "$SUBJECT_ID" \
  --template "$FS_SUBJECT_DIR/mri/aparc+aseg.mgz" \
  --o "$OUTDIR/lh.curv.mgz"

mri_surf2vol \
  --so "$FS_SUBJECT_DIR/surf/rh.white" "$FS_SUBJECT_DIR/surf/rh.curv" \
  --subject "$SUBJECT_ID" --identity "$SUBJECT_ID" \
  --template "$FS_SUBJECT_DIR/mri/aparc+aseg.mgz" \
  --o "$OUTDIR/rh.curv.mgz"
