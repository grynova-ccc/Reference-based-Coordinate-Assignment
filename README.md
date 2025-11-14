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

## ⚠️ Limitations of the Method


## 🔧 Getting Started With the Code

## Contact
If you have questions, feel free to reach out: stivllenga@gmail.com

## Citation
If you use this project in your research, please cite:
URL URL URL
