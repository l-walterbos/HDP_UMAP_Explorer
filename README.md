# Interactive UMAP explorer
This interactive platform for exploring [UMAP](https://umap-learn.readthedocs.io/en/latest/) projections is part of our publication [INSERT FINAL TITLE HERE]. For detailed information on the generation and processing of the data we refer you to our paper [INSERT LINK] (open access). **In short:** we performed high-throughput simulations on Halide Double Perovskites (HDPs) (chemical formula A2BB'X6) and here we tried to use UMAP on this data to see whether we can reveal trends underlying the electronic-structure data. 

This explorer is based on [Dash](http://dash.plotly.com) and allows you to switch between several precomputed UMAP projections (contained in Precomputed_UMAPprojections_*.csv). Our UMAP projections are based on projected Density of States and COHP curves, which we make sure are on equal energy axes, and are presented to the UMAP algorithm in several variations. The explorer also offers different values to color the UMAP projection with (like bandgaps, net spin, or lattice constants). Furthermore, clicking a point from the projections, shows some important simulated quantities from that specific composition and shows the corresponding DOS and COHP/COBI plots. There is also a periodic table plot, colored by some averaged values based on the occurances of the elements as one of the B-sites in the dataset. 

## Running the explorer
Running the explorer is very straight forward. Setup a new python environment (**python==3.13**) and install the packages from requirements.txt
Then the application can be run from the terminal 
```
python app.py
```
The app will then be running on local host and you can explore the data.

## Available data and preparation
The parsed pDOS, pCOHP, and pCOBI data is available in json format and included in the subdirectories lsodos_smeared, cohps_smeared, cobis_smeared. These are used as the basis for the UMAP projections, but are also included since they are needed for the supplementary figures displayed underneath the UMAP projection plot. All directories contain the word `smeared` because we added some additional Gaussian smearing (0.05 eV) to smooth out the sharpest peaks in the data a bit.

There is also the csv 'HDP_CombinedInfo_260419.csv'. This csv is read as a pandas DataFrame and provides the data used for coloring the UMAP projection and the periodic table plot. The index of this csv (index_col=0) is used to access the correct data for each composition across all plots. 
(The index is CompID, structrured as XXXX_CsBB'X with XXXX a unique number assigned starting from 1000 and B, B', and X varying for each composition). 
For the precise content of the CombinedInfo DataFrame see the paper linked above.

To turn the pDOS and pCOHP into UMAP projections, we first need to assure all input data is equalized. VASP and LOBSTER calculations 
let you set the number of energy points, but the energy range varies from composition to composition. To put all curves back onto an equal footing, we use a binning or fingerprinting method. This method sums all DOS/COHP contributions from the old energy axis that fall within an energy bin in the new axis. This method has been adapted from the works:
- Purcell et al., 2023, 10.1038/s41524-023-01063-y
- Kuban et al., 2024, 10.1039/D4DD00258J
The functionality for this all contained in `dos_dataprepare.py` (despite the name the pCOHP curves are also binned in this script). Besides fingerprinting the DOS/COHP, the script also limits the parts of the curves that is included. In our case we only include 5 eV below the Valence Band Maximum (VBM) and 5 eV above the Conduction Band Maximum (CBM). 
So the new energy axis will run from -5 to (Max(CBM) +5), but for each individual composition only contributions from -5 to its own CBM+5 are included in the fingerprint. Any values outside that range will be 0. 
The DOS fingerprints are saved for the total, and projected to each site (A/B1/B2/X), and for spin-up and spin-down separately. The pCOHP are available for the 2 times 6 B-X bonds; these are fingerprinted for the 6 bond average, and averaged over 2 bonds aligned with each cartesean axis, again also spin-up and spin-down separated. All these fingerprints are saved to the smeared_histogrammed_* subdirectories. The fingerprints are saved as csv's, with the index being the center of the energy bins, and the column_heads being the CompID's. They are saved separately to make it easy to try different combinations of the data. 

## UMAP projections
Precomputing the UMAP projections is done in `umap_preparation.py`. We read in the different fingerprinted data, and define a dictionary of datasets to project. The keys of this dictionary are used to represent the dataset, also in the interactive viewer. The value in the datadictionary must be `pd.DataFrame` or a `list[pd.DataFrame]`. If it is a list of DataFrames, the code will project the DataFrames individually first and then combine the projections using the intersection operator see [UMAP docs](https://umap-learn.readthedocs.io/en/latest/composing_models.html) for more info. Currently the combination of different projections is pretty hardcoded in, so only list[DataFrame] with lengths of 2 and 4 are supported, and the combination operator cannot be varied. To describe the datasets we defined, we use [something, something] to indicate that we concatonated these DataFrames (using ignore_index=True); _up and _down indicate which spin-channel; * between to datasets indicate that these are intersected. The datasets we projected then are:
- TDOS: [tdos_up, tdos_down]
- B-site pDOS: [pDOS(B1)_up, pDOS(B1)_down] * [pDOS(B2)_up, pDOS(B2)_down]
- Alternative B-site pDOS: [pDOS(B1)_up, pDOS(B2)_up] * [pDOS(B1)_down, pDOS(B2)_down]
- Separated B-site pDOS: [pDOS(B1)_up] * [pDOS(B1)_down] * [pDOS(B2)_up] * [pDOS(B2)_down]
- (6 bond) Average B-X pCOHP: [pCOHP(B1-X)_up, pCOHP(B1-X)_down] * [pCOHP(B2-X)_up, pCOHP(B2-X)_down]
We also have for each dataset a normalized alternative (indicated with the prefix _norm._), where the area from each fingerprint is set to equal 1.

Secondly, we define a list of metrics and nearest-neighbor values we want to use to generate projections. In our case the metric list comprises of:
- euclidean 
- seuclidean (euclidean scaled with inverse standard deviation)
- manhattan
- wminkowski (in our case manhattan scaled with inverse standard deviation)
- bray-curtis (sum(|a_i-b_i|)/sum(|a_i+b_i|))
- cosine
And we used nearest-neighbor values: [5,15,25,50,100]. UMAP has an additional parameter min_dist, which we have set equal to 0.1 for all projections.
The script then automatically performs the UMAP projection for each dataset, metric, nn_value combination and saves the results to `Precomputed_UMAPprojections_*.csv`. The csv has CompID as the index (the input data is transposed when fed into the UMAP algorithm) and a MultiIndex.from_product([list[datasetnames],list[metrics],list[nn_values],['x','y']]) for the columns.

## Adding or adapting projections
The interactive viewer reads the Precomputed_UMAPprojections csv in as a DataFrame again, and automatically puts the datasetnames into a Dropdown select, the metrics into a Radio select, and nn_values into a Slider. If you are interested in making alternative projections, it should be straight forward to do so. If you stay within the UMAP framework it should be possible to just add your data and/or metrics to the datadict and metric_list. 

You can also just perform any projection how you wish, as long as you maintain the 4-layer MultiColumn with ('x','y') as the lowest layer, and a numerical value for the nn_value layer (even if it's just a single value). This means that the interactive viewer can be used to view the projection from any Dimensionality Reduction method. As long as the Index and MultiColumn structure is maintained this should be possible without any changes to the app code.

If you are interested in using the viewer code for a dataset, it should still be relatively easy to adapt as long as you have some CompiledData DataFrame where the index is also used as a prefix for other files. Although, naming convention and differences in parsing might incite some errors.