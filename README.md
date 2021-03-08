[![GitHub license](https://img.shields.io/github/license/martibosch/jonction-tree-canopy.svg)](https://github.com/martibosch/jonction-tree-canopy/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4589559.svg)](https://doi.org/10.5281/zenodo.4589559)

# Jonction tree canopy

Tree canopy map for the Jonction neighbourhood (Geneva) at the 40cm resolution obtained with [DetecTree](https://github.com/martibosch/detectree) [1] from Geneva's [Orthophoto 2019](https://ge.ch/sitg/fiche/3137) and [LiDAR 2019](https://ge.ch/sitg/fiche/1827) datasets. The obtained tree canopy raster file (22.0 MB) [can be downloaded from Zenodo](https://doi.org/10.5281/zenodo.4589559).

## Technical specifications

* **Source**: Geneva's [Orthophoto 2019](https://ge.ch/sitg/fiche/3137) and [LiDAR 2019](https://ge.ch/sitg/fiche/1827) datasets
* **CRS**: CH1903+/LV95 -- Swiss CH1903+/LV95 ([EPSG:2056](https://epsg.io/2056))
* **Resolution**: 40cm
* **Extent**: From file [agglom-extent.shp](https://github.com/martibosch/jonction-tree-canopy/blob/master/data/raw/jonctioncarte.tif).
* **Method**: supervised learning (AdaBoost) with 2 classifiers on manually-generated ground truth masks for 2 training tiles (out of a total 65 tiles) of 512x512 pixels. See Yang et al. [2] for more details.
* **Accuracy**: 94.88%, estimated from a ground truth mask for 1 tile of 512x512 pixels (computed from LiDAR data).

## Citation

If you use this dataset, a citation to DetecTree would certainly be appreciated. Note that DetecTree is based on the methods of Yang et al. [2], therefore it seems fair to reference their work too. Additionally, the sources, i.e., Geneva's [Orthophoto 2019](https://ge.ch/sitg/fiche/3137)  and [LiDAR 2019](https://ge.ch/sitg/fiche/1827) datasets can be acknowledged. An example citation in an academic paper might read as follows:

> The tree canopy dataset for the Jonction neighbourhood (Geneva) has been obtained from the Geneva's Orthophoto 2019 and LiDAR 2019 datasets with the Python library DetecTree (Bosch, 2020), which is based on the approach of Yang et al. (2009).

## Steps to reproduce

1. Create a conda environment

```bash
make create_environment
```

2. Activate it

```bash
conda activate jonction-tree-canopy
```

3. Download the train/test split used in this workflow run from Zenodo<sup>[1](#note-1)</sup>.

```bash
make download_zenodo_split
```
  
4. Compute the responses for the training tiles (i.e., ground truth masks) from LiDAR data (the response tiles will be automatically dumped to `data/interim/response-tiles`):

```bash
make response_tiles
```

5. Train tree/non-tree classifiers:

```bash
make train_classifiers
```

6. Use the classifiers to predict a tree canopy raster:

```bash
make tree_canopy
```

**Optional**: If you want to use this environment in Jupyter, you might register its IPython kernel as follows:

```bash
make register_ipykernel
```

## Notes

1. <a name="note-1"></a> If you want to execute your own run of this workflow, you might skip the `make download_zenodo_split` command from the third step and run `make train_test_split` instead. Note however that such a step of the workflow has a stochastic component (i.e., from the [K-Means initialization](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)), and therefore you might obtain a different train/test split which might require you to [download LiDAR data manually from SITG (link in French)](https://www.etat.ge.ch/geoportail/pro/?method=showextractpanel).

## Acknowledgments

* With the support of the École Polytechnique Fédérale de Lausanne (EPFL)
* Project based on [Henk Griffioen's version](https://github.com/hgrif/cookiecutter-ds-python) of the [cookiecutter data science project template](https://drivendata.github.io/cookiecutter-data-science). #cookiecutterdatascience

## References

1. Bosch, M. (2020). Detectree: Tree detection from aerial imagery in Python. Journal of Open Source  Software (under review).
2. Yang, L., Wu, X., Praun, E., & Ma, X. (2009). Tree detection from aerial imagery. In Proceedings of the 17th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (pp. 131-137). ACM.
