![](caravan-long-logo.png)

# Caravan - A global community dataset for large-sample hydrology

This repository is part of the Caravan project/dataset.

## What is Caravan

_Caravan_ is an open community dataset of meteorological forcing data, catchment attributes, and discharge data for catchments around the world. Additionally, Caravan provides code to derive meteorological forcing data and catchment attributes in the cloud, making it easy for anyone to extend Caravan to new catchments. The vision of Caravan is to provide the foundation for a truly global open source community resource that will grow over time. 

The Caravan dataset that was released together with the [paper](https://www.nature.com/articles/s41597-023-01975-w) and can be found [here](https://doi.org/10.5281/zenodo.10968468).

> [!IMPORTANT]
> The current Caravan version is v1.5 in which we added Penman-Monteith PET, given the known issues with the potential_evaporation band from ERA5-Land. We also added PET related climate indices derived with the new Penman-Monteith band. For consistency, the old ERA5-Land potential evaporation band and the old climate indices were kept but renamed. 

> [!TIP]
> Join the [Caravan Google Groups](https://groups.google.com/g/caravan-dataset) to get email notifications for version updates and other important announcement around the Caravan community dataset.


## Caravan MultiMet

We recently released the [Caravan MultiMet extension](https://arxiv.org/abs/2411.09459), which adds several weather products for all Caravan basins (including all basins from all extensions). The MultiMet extension contains several nowcast products (CPC, IMERG v07 Early, CHIRPS), as well as three different  forecast products (ECMWF IFS HRES, GraphCast, CHIRPS-GEFS) with multiple bands and 10 days (IFS, GraphCast) or 16 day (CHIRPS-GEFS) lead times. The data is saved in zarr files that can be downloaded from Zenodo and is also being hosted on public GCP bucket. For details check the paper [Caravan MultiMet extension](https://arxiv.org/abs/2411.09459) and the [notebook](examples/Caravan_MultiMet_Extending_Caravan_with_Multiple_Weather_Nowcasts_and_Forecasts.ipynb) that can be executed on Colab and examplifies how to work with the data from the GCP bucket.

**Note**: In contrast to the original Caravan data, all data for all basins in the MultiMet extension is in UTC-0, which is the original timezone for all of these global weather products. Since not all are available in sub-daily resolution, and hence data can't be shifted easily to local time, we decided to keep all data for all basins in UTC-0. For that reason, the MultiMet extension also contains ERA5-Land reanalysis data and here, also in UTC-0. 

## About this repository

The purpose of this repository is twofold:

1. It contains the code  
    - that was used to derive all of the data included in Caravan, and 
    - that is required to extend Caravan to any new location for free in the cloud.
2. It acts as a community hub (see [discussion forum](https://github.com/kratzert/caravan/discussions)) to
    - share news and updates on Caravan,
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
@article{kratzert2023caravan,
  title={Caravan-A global community dataset for large-sample hydrology},
  author={Kratzert, Frederik and Nearing, Grey and Addor, Nans and Erickson, Tyler and Gauch, Martin and Gilon, Oren and Gudmundsson, Lukas and Hassidim, Avinatan and Klotz, Daniel and Nevo, Sella and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={61},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

Additionally, we would highly appreciated if you also cite the corresponding manuscripts of the source datasets. For details on the references, see the information included in the licenses folder of the [Caravan dataset](https://doi.org/10.5281/zenodo.10968468)

## Contact

If you have any questions/feedback regarding the Caravan dataset/project, please contact Frederik Kratzert kratzert(at)google.com
