# RNA reverse folding

Background: RNA plays a crucial role in cellular life activities, from gene expression regulation to catalyzing biochemical reactions, all of which rely on the participation of RNA. RNA folding refers to the process by which RNA sequences spontaneously form specific three-dimensional structures. However, RNA reverse folding is a more challenging problem.  In [the 3rd World Science Intelligence Competition](http://competition.sais.com.cn/competitionDetail/532314/format), the core of this competition is the RNA reverse folding problem, which involves generating one or more RNA sequences based on a given RNA three-dimensional skeleton structure, so that these sequences can fold and approach the given target three-dimensional skeleton structure as closely as possible. The evaluation criterion is the recovery rate of the sequence.

This repository provides a hybrid sequence generation network based on message passing neural networks and Transformer blocks. The recovery rate on hundreds of publicly available RNA backbone structures is 67.34%.

## Installation
**Step 1: Clone this repository:**
```bash
git clone https://github.com/prison68/RNA_reverse_folding.git
cd RNA_reverse_folding
```
**Step 2: Environment Setup:**

Create and activate a new conda environment.

```bash
conda create -n RNARF
conda activate RNARF
```
**Step 3: Install Dependencies:**

Install the required dependencies with the supported versions.
```
pip install -r requirements.txt
```

## Dataset
In `RNAdesignv2/train`, is the official dataset provided by the competition(only uploaded a portion as an example):

- Input format, including RNA backbone atoms and RNA side chain atoms:

  - P   (phosphate group)
  - O5' (5'oxygen atom)
  - C5' (5'carbon atom)
  - C4' (4'carbon atom)
  - C3' (3'carbon atom)
  - O3' (3'oxygen atom)
  - N1(pyrimidine base) or N9(purine base)
  
The input data is provided in numpy array format, where each atom are three-dimensional coordinates (x, y, z). If an atom does not exist in the original data, that position will be filled with NaN.

- Output format, provided in FASTA format file. The FASTA file consists of two parts:

  - Sequence identifier line: starts with`>`, followed by the unique identifier of the sequence.
  - Sequence line: The bases (A, U, C, G) of the RNA sequence are arranged in order.

For example, it should be like this:
```text
>1A9N_1_Q
CCUGGUAUUGCAGUACCUCCAGGU
```

## Usage

### Training
We provide the [main.py](./main.py) script for train, example usage:

```
python main.py --batch_size 16 --epochs 200 --lr 1e-4
```
The full list of options is in [parser.py](./parser.py).

### Inference

We provide the [run.py](./run.py) script for inference, example usage:

```
python run.py
```

Notes:

- When inferencing, read input data from path `./saisdata`
- All generated FASTA sequences will be saved to the `./saisresult/submit.csv`.

## Acknowledgements
This code is builds on the code from the [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), [RDesign](https://github.com/A4Bio/RDesign) and [sais_third_medicine_baseline](https://www.modelscope.cn/datasets/Datawhale/sais_third_medicine_baseline) codebase.
