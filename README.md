These are the scripts I used to make the UMAP interactieve explorer app

CsBBX_Analyzer.py is the script I used for parsing the data from vasp into more easily usable format
It consists of 2 classes 'SingleHDPanalysis' which contains the functions to get the data for a single composition
'GroupedAnalysis' has the function to select right basis (for comps with spilling >3% it switches) for the rest it has 
process_xxxxx which is a single call to the relevant SingleHDPanalysis function, and then get_xxxx_data calls this function with MultiProcessing for speedup.
Directly relevant for the UMAP projection are mostly:
	SingleHDPanalysis.get_tdos() (line 502)
	SingleHDPanalysis.get_tdos_per_site() (line 521)
	SingleHDPanalysis.get_spddos_per_site() (line 605) 
	SingleHDPanalysis.get_parsed_coxx() (line 1192)


dos_dataprepare.py is the script where I interpolate the DOS all onto the same energy axis and from which I have generated the histogrammed_Xdos_xx.csv 's.
I generated all separate csv files to make it easy to try different combinations of UMAP projections. The csv files are structured as follow:
	Columnheads are the composition names
	the rows correspond to the different energy values
	index_col=0, there the energy values are contained
	Only 5eV is included around VBM and CBM, as contained in the HDP_CombinedInfo_281025.csv overview csv
	These values were extracted from the DOSCAR.LSO.lobster files, but due to vasp inconsistencies VBM != 0 in all cases, so this is made right in the interpolation scheme


HDP_PlotslyPlots.py contains the functions to plot the DOS and COXX. These functions only take as input a comp_id (e.g. 1003_CsAgDyCl) this the index of HDP_CombinedInfo.csv,
and the CombinedInfo DataFrame. From there it will read the relevant json.gz and return the plot (with also energy shifted so VBM=0 and making sure the majority spin channel = spin.up)

umap_prepartation.py is where I prepare several UMAP projections which can easily be switched between in the app. These specific projections are not final yet.
It just precomputes the x,y values for each compoisiton based on each projection and then saves that to PrecomputedUMAPprojections_{date}.csv with a MultiIndex as column header (('projectionname','x')('projectionname','y'))

app.py is where all the code exists for the Dash app. THere is a Dropdown for selection UMAP projection, based on the level 0 column names in PrecomputedUMAPprojections. A subset of data from CombinedInfo.csv
is put into 'dplot'. All of the data contained in dplot can be used to color the UMAP projection by. THere is the option to switch between linear and log2 scaling. There is a periodic table with the 
counts of different elements being featured in a composition, but there is no callbacks yet with this subfigure.
There are the DOS COHP-B1 COHP-B2 plots. and some basic information on a composition. These values get updated each time you click a point in the UMAP projection

umap_plot.py was earlier testing of mine with umap and not relevant for the app