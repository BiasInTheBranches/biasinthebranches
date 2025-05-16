# harness
**Bias in the Branches: A Benchmark on Phylogenetic Representation Gaps in Language Models**

Abstract:
> Protein and genome language models are trained on corpora that are highly taxonomically imbalanced. Yet, benchmarks seldom quantify how that imbalance affects the representations they learn. 
To address this, we introduce a novel benchmark with both amino acid and nucleotide-level sequences that systematically pairs representatives from common and rare taxonomic families across all domains of life. 
By maintaining identical label distributions between paired sets, we isolate taxonomic effects and test three biological tasks: homology detection, function annotation, and structural fold classification. 
Our analysis reveals significant bias patterns within and across domains, with differences in performance up to 24% between common and rare taxa, and varies systematically by task type.
We demonstrate directional asymmetry in cross-taxonomic transfer, where models trained on rare taxa generalize better to common taxa for homology and function, while structure classification shows the reverse pattern. 
These results indicate model architecture and dataset curation prove more influential than parameter count for taxonomic generalization. 
This work provides both metrics and insights for developing more inclusive and generalizable foundation models for computational biology. 


## Installation

To install the package, clone and run the following command:
```bash
pip install -U pip setuptools wheel
pip install '.[all]'
```

## Usage
To run a benchmark, pass the appropriate YAML config file to this command:
```bash
python -m harness.evaluate --config [PATH]
```

To run the full benchmark suite, use the following command:
```bash
# Optionally set HF_HOME to a custom directory to store the datasets
# export HF_HOME=/path/to/dir
bash scripts/run-multi.sh
```

To run the reporting script (generates figures and tables), use the following command:
```bash
python -m harness.report [INPUT_DIR] [OUTPUT_DIR]
```

If you have run the scripts above, you can use the following command to generate a report:
```bash
python -m harness.report results/ figures/
```

## Data
### Downloading
Data is hosted on [Hugging Face](https://huggingface.co/datasets/biasinthebranches/uniprot-taxonomy-splits). You can download the datasets using the following command:
```bash
export HF_HOME=/path/to/dir
huggingface-cli download biasinthebranches/uniprot-taxonomy-splits --repo-type=dataset
```

This will download the datasets to wherever you set `HF_HOME`. You can then set the `HF_HOME` environment variable to point to allow HF to find the datasets. This step is _NOT_ required, these datasets will be automatically downloaded when you run the benchmark. However, you get get them ahead of time this way.


### Loading/Inspecting
You can load a dataset, the following command:
```python
from datasets import load_dataset
config_name = 'nokingdom-phylum-top0.80-pfam'
dataset = load_dataset("biasinthebranches/uniprot-taxonomy-splits", config_name)
```

The available config options are one of:
```
kingdom: nokingdom, archaea, bacteria, eukaryota, viruses
taxa_level: phylum, class, order, family, genus  # note, archaea do not have phylum
split: top0.80, bottom0.20
label: pfam, ec0, ec1, ec2, ec3, gene3d0, gene3d1, gene3d2, gene3d3
```


