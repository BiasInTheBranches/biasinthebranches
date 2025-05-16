#!/bin/bash

CONFIG_FILES=(

    "../configs/nokingdom/calm.yaml"
    "../configs/nokingdom/dnabert2.yaml"
    "../configs/nokingdom/esm-150m.yaml"
    "../configs/nokingdom/esm-650m.yaml"
    "../configs/nokingdom/esmc-300m.yaml"
    "../configs/nokingdom/esmc-600m.yaml"
    "../configs/nokingdom/nucleotidetransformer.yaml"
    "../configs/nokingdom/ngram-aa.yaml"
    "../configs/nokingdom/ngram-dna.yaml"
    "../configs/nokingdom/onehot-aa.yaml"
    "../configs/nokingdom/onehot-dna.yaml"
    "../configs/nokingdom/ankh-base.yaml"

    "../configs/bacteria/calm.yaml"
    "../configs/bacteria/dnabert2.yaml"
    "../configs/bacteria/esm-150m.yaml"
    "../configs/bacteria/esm-650m.yaml"
    "../configs/bacteria/esmc-300m.yaml"
    "../configs/bacteria/esmc-600m.yaml"
    "../configs/bacteria/nucleotidetransformer.yaml"
    "../configs/bacteria/ngram-aa.yaml"
    "../configs/bacteria/ngram-dna.yaml"
    "../configs/bacteria/onehot-aa.yaml"
    "../configs/bacteria/onehot-dna.yaml"
    "../configs/bacteria/ankh-base.yaml"


    "../configs/eukarya/calm.yaml"
    "../configs/eukarya/dnabert2.yaml"
    "../configs/eukarya/esm-150m.yaml"
    "../configs/eukarya/esm-650m.yaml"
    "../configs/eukarya/esmc-300m.yaml"
    "../configs/eukarya/esmc-600m.yaml"
    "../configs/eukarya/nucleotidetransformer.yaml"
    "../configs/eukarya/ngram-aa.yaml"
    "../configs/eukarya/ngram-dna.yaml"
    "../configs/eukarya/onehot-aa.yaml"
    "../configs/eukarya/onehot-dna.yaml"
    "../configs/eukarya/ankh-base.yaml"


    "../configs/archaea/calm.yaml"
    "../configs/archaea/dnabert2.yaml"
    "../configs/archaea/esm-150m.yaml"
    "../configs/archaea/esm-650m.yaml"
    "../configs/archaea/esmc-300m.yaml"
    "../configs/archaea/esmc-600m.yaml"
    "../configs/archaea/nucleotidetransformer.yaml"
    "../configs/archaea/ngram-aa.yaml"
    "../configs/archaea/ngram-dna.yaml"
    "../configs/archaea/onehot-aa.yaml"
    "../configs/archaea/onehot-dna.yaml"
    "../configs/archaea/ankh-base.yaml"


    "../configs/viruses/calm.yaml"
    "../configs/viruses/dnabert2.yaml"
    "../configs/viruses/esm-150m.yaml"
    "../configs/viruses/esm-650m.yaml"
    "../configs/viruses/esmc-300m.yaml"
    "../configs/viruses/esmc-600m.yaml"
    "../configs/viruses/nucleotidetransformer.yaml"
    "../configs/viruses/ngram-aa.yaml"
    "../configs/viruses/ngram-dna.yaml"
    "../configs/viruses/onehot-aa.yaml"
    "../configs/viruses/onehot-dna.yaml"
    "../configs/viruses/ankh-base.yaml"
)

for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "Running evaluation for $CONFIG..."
    python -m harness.evaluate --config "$CONFIG"
    echo "Finished evaluation for $CONFIG"
    echo "--------------------------------------"
done