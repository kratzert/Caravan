![](caravan-long-logo.png)

# Caravan - A global community dataset for large-sample hydrology

This repository is part of the Caravan project/dataset.

## What is Caravan

_Caravan_ is an open community dataset of meteorological forcing data, catchment attributes, and discharge data for catchments around the world. Additionally, Caravan provides code to derive meteorological forcing data and catchment attributes in the cloud, making it easy for anyone to extend Caravan to new catchments. The vision of Caravan is to provide the foundation for a truly global open source community resource that will grow over time. 

The Caravan dataset that was released together with the [paper](https://www.nature.com/articles/s41597-023-01975-w) and can be found [here](https://doi.org/10.5281/zenodo.7387919).

## About this repository

The purpose of this repository is twofold:

1. It contains the code  
    - that was used to derive all of the data included in Caravan, and 
    - that is required to extend Caravan to any new location for free in the cloud.
2. It acts as a community hub (see [discussion forum](https://github.com/kratzert/caravan/discussions)) to
    - share news and updates on Caravan ),
    - for anyone to share extensions of Caravan to new regions.

See ["Extend Caravan"](https://github.com/kratzert/Caravan/wiki/Extending-Caravan-with-new-basins) for a detailed description about how to extend Caravan to any new region/basin with the code provided in this repository. See ["How to contribute"](#how-to-contribute) for more details about how to contribute to the Caravan project.

## How to contribute

Our main vision with Caravan is that this dataset will grow over time. Anyone, with as little as streamflow records and catchment boundaries of one (or more) basins, can contribute to extending the Caravan dataset to new regions. The code provided in this dataset can be used to:

1. Compute static catchment attributes on Google Earth Engine.
2. Compute time series of spatially-averaged meteorological forcings on Google Earth Engine.
3. Postprocess the Earth Engine outputs locally and to combine it with streamflow, as well as to compute some additional climate indices.

The generated output is already in a folder structure that can be easily integrated into the existing dataset. Additionally, every data that is contributed contains a separate license/info file, attributing your contribution to this project and explaining the source of license specification of this addition. Follow [this guide](https://github.com/kratzert/Caravan/wiki/Sharing-New-Data) for more information on how to share your data with the community.

## Reference

The Caravan dataset (and the corresponding manuscript) are currently under revisions. If you use the Caravan dataset in your research/work, the recommended citation is:

```bib
@article{kratzert2022caravan,
  title={Caravan - A global community dataset for large-sample hydrology},
  author={Kratzert, Frederik and Nearing, Grey and Addor, Nans and Erickson, Tyler and Gauch, Martin and Gilon, Oren and Gudmundsson, Lukas and Hassidim, Avinatan and Klotz, Daniel and Nevo, Sella and Shalev, Guy and Matias, Yossi},
  year={2022},
  publisher={EarthArxiv},
  doi={10.31223/X50S70}
}
```

Additionally, we would highly appreciated if you also cite the corresponding manuscripts of the source datasets. For details on the references, see the information included in the licenses folder of the [Caravan dataset](https://doi.org/10.5281/zenodo.7387919)

## Contact

If you have any questions/feedback regarding the Caravan dataset/project, please contact Frederik Kratzert kratzert(at)google.com
