# Navigating the chemical space with a molecular GPS
<p align="center">
  <img src="Images/image_1.png" width="650" />
</p>

## Welcome! 👋
This repository accompanies the research paper of the same name and provides an implementation of Reference-based Coordinate Assignment (RCA) — a method for projecting high-dimensional data into a space of different dimensionality.

RCA treats points in the original high-dimensional space as belonging to two groups:

* X: a set of data points defined by the user (e.g., from a dataset)

* Y: a set of reference points that aren't part of the original dataset

The core idea of RCA is to project the X space into a new coordinate system while preserving only the pairwise distances between Y–Y and X–Y points. This approach is inspired by the concept of multilateration. As an analogy, you can think of the Y points (called reference points) as cell towers or satellites, and the X points as unknown locations. RCA aims to determine the positions of the X points using only their distances to the Y reference points.

## ❓ Why This Project?
In the introduction of the article, we review several existing techniques for dimensionality reduction. However, RCA is best understood not as a traditional dimensionality-reduction method, but as a projection technique: it projects the original input space into a new space whose dimensionality equals the number of reference points.

The motivation behind developing RCA is twofold. First, it provides an intuitive framework that can be directly compared to GPS and multilateration-based tracking systems—except now applied in high-dimensional spaces. Second, RCA offers a quantitative way to express the “similar structure → similar property” principle. Across multiple datasets, machine-learning tasks, and model architectures, we show that using RCA-projected spaces as input not only reduces computational cost and runtime, but also consistently improves predictive accuracy.

## 🔧 Installation

To generate the RCA space from the input space, install the package using:

```bash
pip install RCA-space
```
The core function for generating the RCA space, `RCA_vectorised`, depends only on [NumPy](https://numpy.org/). However, to generate the required inputs for this function, you may use `RCA_reference_projection`, which introduces additional dependencies on both [NumPy](https://numpy.org/) and [scikit-learn](https://scikit-learn.org/stable/).

## 🚀 How to Use

Producing the representation for a single molecule is not very useful when training a machine-learning model. However, when using the one_electron_matrices package, only one CIF file is needed to compute its corresponding representation. To generate representations for multiple CIF files in parallel, you can use the following code:
```python
from joblib import Parallel, delayed
from pathlib import Path
from your_module import oem_rep  # import your function

cif_folder = Path("PATH_TO_CIFS")
output_folder = Path("OUTPUT_PATH")
output_folder.mkdir(exist_ok=True, parents=True)

cif_files = sorted(cif_folder.glob("*.cif"))

N = N  # adjust CPU usage

Parallel(n_jobs=N)(
    delayed(oem_rep)(str(cif_path), output_path=str(output_folder))
    for cif_path in cif_files)
```

## Contact
If you have questions, feel free to reach out: stivllenga@gmail.com

## Citation
If you use this project in your research, please cite:
URL URL URL
