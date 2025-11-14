# Navigating the chemical space with a molecular GPS
<p align="center">
  <img src="Images/image_1.png" width="650" />
</p>

## Welcome! 👋
This repository accompanies the research paper of the same name and provides an implementation of Reference-based Coordinate Assignment (RCA) — a method for projecting high-dimensional data into a space of different dimensionality.

RCA treats points in the original high-dimensional space as belonging to two groups:

> X: a set of data points defined by the user (e.g., from a dataset)

> Y: a set of reference points that aren't part of the original dataset

The core idea of RCA is to project the X space into a new coordinate system while preserving only the pairwise distances between Y–Y and X–Y points. This approach is inspired by the concept of multilateration. As an analogy, you can think of the Y points as cell towers or satellites, and the X points as unknown locations. RCA aims to determine the positions of the X points using only their distances to the Y reference points.
