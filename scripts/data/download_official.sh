#!/usr/bin/env bash
# Fetch official neural-operator benchmarks. Resumable; safe to re-run.
#   DATA_ROOT  destination (default: <repo>/datasets/raw, the path sgpo/data/benchmarks.py reads; override with LOGOS_RAW)
#   GDOWN      gdown binary
# Sources: de Hoop et al. (Caltech), neuraloperator (Zenodo),
#          FNO and Geo-FNO author folders (Google Drive; quota-limited, so retried hourly).
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
DATA_ROOT=${DATA_ROOT:-$HERE/datasets/raw}
GDOWN=${GDOWN:-gdown}

zenodo() {  # record file
    mkdir -p "$DATA_ROOT/zenodo"
    curl -sSL -C - --retry 8 -o "$DATA_ROOT/zenodo/$2" \
        "https://zenodo.org/api/records/$1/files/$2/content"
}

hf() {  # repo path: Hugging Face mirror of the FNO authors' files (see data/data.md)
    mkdir -p "$DATA_ROOT/hf_mutransfer"
    curl -sSL -C - --retry 8 -o "$DATA_ROOT/hf_mutransfer/$(basename "$2")" "https://huggingface.co/datasets/$1/resolve/main/$2"
}

zen_all() {
    zenodo 12825163 nsforcing_1024.tgz
    for f in NavierStokes_inputs NavierStokes_outputs; do
        mkdir -p "$DATA_ROOT/caltech"
        curl -sSL -C - --retry 8 -o "$DATA_ROOT/caltech/$f.npy" "https://data.caltech.edu/api/records/fp3ds-kej20/files/$f.npy/content"
    done
    hf LDA1020/muTransfer-FNO-data darcy/piececonst_r421_N1024_smooth1.mat
    hf LDA1020/muTransfer-FNO-data burgers/burgers_data_R10.mat
    mkdir -p "$DATA_ROOT/fno/burgers" && mv -n "$DATA_ROOT/hf_mutransfer/burgers_data_R10.mat" "$DATA_ROOT/fno/burgers/" 2>/dev/null
    echo ZENODO_DONE
}

drive_files=(  # id name
    "1Z1uxG9R8AdAGJprG5STcphysjm56_0Jf Darcy_421.zip"
)
drive_folders=(1cznHmQO-hB_VlWOfh7IFpjsPVOYyghGJ 1_pGrwS6rwzhyaCXS6MEmmPKWFXNeRnhj
               1iOgLyfqGtXuZ25KHsDWIh_ZJAfO_HxMO 1lo7n8dWtpcVhhaEpQavh9UoSkMOs1UdC)

drive_loop() {
    mkdir -p "$DATA_ROOT/fno" "$DATA_ROOT/geofno"
    for attempt in $(seq 1 24); do
        missing=0
        for e in "${drive_files[@]}"; do
            set -- $e
            [ -s "$DATA_ROOT/fno/$2" ] && continue
            (cd "$DATA_ROOT/fno" && "$GDOWN" "$1" -O "$2") || { rm -f "$DATA_ROOT/fno/$2"; missing=1; }
        done
        for id in "${drive_folders[@]}"; do
            (cd "$DATA_ROOT/geofno" && "$GDOWN" --folder --continue "$id") >/dev/null 2>&1 || missing=1
        done
        [ "$missing" = 0 ] && { echo DRIVE_DONE; return 0; }
        echo "attempt $attempt: quota, sleeping 1h"; sleep 3600
    done
    echo DRIVE_GAVE_UP
}

case "${1:-all}" in
    zenodo) zen_all ;;
    drive)  drive_loop ;;
    all)    zen_all & drive_loop & wait ;;
esac
