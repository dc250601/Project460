<h1 align="center">
Detection of Dual Active Galactic Nuclei using Self-Supervised Active Learning Guided Detection System</br>
(SSALD)
</h1>
<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
[![Open Source Love](https://firstcontributions.github.io/open-source-badges/badges/open-source-v2/open-source.svg)](https://github.com/firstcontributions/open-source-badges)

Official Repository for Project460.</br>
The main purpose of this project
is to explore the use of self-supervised and active learning based algorithms to classify Dual-Active Galactic Nuclei from using a minuscle dataset.
The number of labeled samples are less than 100 for this project. The project has access to a extemely large unlabelled dataset SDSS(208 Million).</br>
</div>
<p align="center"><img alt="DAGN" src="images/DAGN.jpg" height = 400 ></p>

### Pipelines
The number of samples in SDSS Dataset is extremely small. Labelling new samples is extremely difficult in our case due to the fact that DAGNs are rare in nature.
Hence we use the following algorithm to train our model. The below pipeline is a mixture Active Learning and Self-Supervised Learning.
```mermaid
  flowchart TD
  A0[Entire SDSS Dataset]-- Self-Supervision ---B0[Feature-Extractor Backbone];
  B0 -->C0[Data Pool: Sampled + Original];
  C0-- Fine tune the Backbone --->D0[KNN-clustering Algorithm]
  D0-- Find best </br> cluster size ---->D0
  D0 -->E0[Sampler]
  E0-- Oracle --->E0
  E0-- Sample=old_samples + new_samples -->C0
  
```
