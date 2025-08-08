#!/usr/bin/env bash
set -eou pipefail
log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage="${1:-0}"
stop_stage="${2:-100}"

REPO_DIR='./data/libriheavy'
FILLER_GZ_DIR='./data/libriheavy/temp'
OUTPUT_DIR='./data/libriheavy-fiiler'

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    mkdir -p "$REPO_DIR"
    if [ ! -d "$REPO_DIR/.git" ]; then
        gh repo clone k2-fsa/libriheavy "$REPO_DIR"
    fi

    cd ./data/libriheavy
    bash run.sh --stage -1 --stop-stage -1
    bash run.sh --stage 1 --stop-stage 2
    cd ../../
fi

if [ ! -d data/libriheavy/cases_and_punc/lhotse/ ]; then
    log "Please refer https://github.com/k2-fsa/libriheavy to download the dataset at ./data/libriheavy"
    exit 255
fi

# Download directory:
#   data/libriheavy
#    ├── cases_and_punc
#    │   ├── kaldi
#    │   └── lhotse
#    │       ├── libriheavy_cuts_dev.jsonl.gz
#    │       ├── libriheavy_cuts_large.jsonl.gz
#    │       ├── libriheavy_cuts_medium.jsonl.gz
#    │       ├── libriheavy_cuts_small.jsonl.gz
#    │       ├── libriheavy_cuts_test_clean.jsonl.gz
#    │       ├── libriheavy_cuts_test_clean_large.jsonl.gz
#    │       ├── libriheavy_cuts_test_other.jsonl.gz
#    │       └── libriheavy_cuts_test_other_large.jsonl.gz
#    └── upper_no_punc

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    mkdir -p "$FILLER_GZ_DIR"
    log "Stage 1: Collect filler-inclusive data."
    for subset in small large dev test_clean test_other test_clean_large test_other_large; do
        python ./data/scripts/collect.py \
            --manifest ./data/libriheavy/cases_and_punc/lhotse/libriheavy_cuts_${subset}.jsonl.gz \
            --subset ${subset} --output_dir $FILLER_GZ_DIR
    done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Stage 2: Extract audio files from tar files"
    for subset in small large dev test_clean test_other test_clean_large test_other_large; do
        tar -xf "./data/libriheavy/download/librilight/${subset}.tar" -C "./data/libriheavy/download/librilight/"
    done

fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    mkdir -p "${OUTPUT_DIR}/audio"
    log "Stage 3: Segment audio files based on filler-inclusive jsonl.gz."
    for subset in small large dev test_clean test_other test_clean_large test_other_large; do
        python ./data/scripts/segment.py \
            --jsonl-gz $FILLER_GZ_DIR/filler_inclusive_${subset}.jsonl.gz \
            --input-dir "./data/libriheavy/" \
            --subset ${subset} --output-dir "${OUTPUT_DIR}/audio"
    done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Stage 4: Normalize text"
    for subset in small large dev test_clean test_other test_clean_large test_other_large; do
        python ./data/scripts/normalize_text.py \
            --jsonl-gz $FILLER_GZ_DIR/filler_inclusive_${subset}.jsonl.gz \
            --subset ${subset} --output-dir "${OUTPUT_DIR}"
    done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Stage 5: Prepare MFA and the pre-trained model"
    if ! command -v mfa >/dev/null 2>&1; then
        log "MFA not found. Please install MFA following:"
        log "https://montreal-forced-aligner.readthedocs.io/en/latest/getting_started.html"
        exit 1
    fi
    log "Download the pre-trained MFA model"
    mfa model download acoustic english_mfa
    mfa model download dictionary english_mfa
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
    log "Stage 6: Extract duration using MFA"
    NPROC=$(nproc)
    for subset in small large dev test_clean test_other test_clean_large test_other_large; do
        mfa align "${OUTPUT_DIR}/audio/${subset}" english_mfa english_mfa "${OUTPUT_DIR}/textgrids/${subset}" -j $NPROC --clean
    done
fi


if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
    log "Stage 7: Extract pitch"
    python ./data/scripts/extract_pitch.py \
        --input-dir "${OUTPUT_DIR}"
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
    log "Stage 8: Classify gender"
    for subset in small large dev test_clean test_other test_clean_large test_other_large; do
        python ./data/scripts/classify_gender.py \
            --input-dir "${OUTPUT_DIR}" --subset "${subset}"
    done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
    log "Stage 9: Integrate features"
    for subset in small large dev test_clean test_other test_clean_large test_other_large; do
        python ./data/scripts/integrate.py \
            --input-dir "${OUTPUT_DIR}" --subset "${subset}"
    done
fi
