import umap
from umap.umap_ import nearest_neighbors

# import umap.plot
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import plotly.express as px
from itertools import product
import warnings
from pathlib import Path
from typing import Literal

warnings.filterwarnings("ignore")
import os

###Location of histogrammed data to use for projections
dos_output_dir = Path(
    "/home/lwalterb/hdp_project/umap_interactive/smeared_histogrammed_doss"
)
cohp_output_dir = Path(
    "/home/lwalterb/hdp_project/umap_interactive/smeared_histogrammed_cohps"
)

dosTup = pd.read_csv(dos_output_dir / "hist_tdos_up.csv", index_col=0)
dosTdown = pd.read_csv(dos_output_dir / "hist_tdos_down.csv", index_col=0)
dosTtot = pd.concat([dosTup, dosTdown], ignore_index=True).T

normdosTup = pd.read_csv(dos_output_dir / "norm_hist_tdos_up.csv", index_col=0)
normdosTdown = pd.read_csv(dos_output_dir / "norm_hist_tdos_down.csv", index_col=0)
normdosTtot = pd.concat([normdosTup, normdosTdown], ignore_index=True).T

dosB1up = pd.read_csv(dos_output_dir / "hist_B1dos_up.csv", index_col=0)
dosB1down = pd.read_csv(dos_output_dir / "hist_B1dos_down.csv", index_col=0)
dosB1tot = pd.concat([dosB1up, dosB1down], ignore_index=True).T

normdosB1up = pd.read_csv(dos_output_dir / "norm_hist_B1dos_up.csv", index_col=0)
normdosB1down = pd.read_csv(dos_output_dir / "norm_hist_B1dos_down.csv", index_col=0)
normdosB1tot = pd.concat([normdosB1up, normdosB1down], ignore_index=True).T

dosB2up = pd.read_csv(dos_output_dir / "hist_B2dos_up.csv", index_col=0)
dosB2down = pd.read_csv(dos_output_dir / "hist_B2dos_down.csv", index_col=0)
dosB2tot = pd.concat([dosB2up, dosB2down], ignore_index=True).T

normdosB2up = pd.read_csv(dos_output_dir / "norm_hist_B2dos_up.csv", index_col=0)
normdosB2down = pd.read_csv(dos_output_dir / "norm_hist_B2dos_down.csv", index_col=0)
normdosB2tot = pd.concat([normdosB2up, normdosB2down], ignore_index=True).T

cohpB1avg_up = pd.read_csv(cohp_output_dir / "hist_cohp_B1avg_up.csv", index_col=0)
cohpB1avg_down = pd.read_csv(cohp_output_dir / "hist_cohp_B1avg_down.csv", index_col=0)
cohpB1avg_tot = pd.concat([cohpB1avg_up, cohpB1avg_down], ignore_index=True).T

cohpB2avg_up = pd.read_csv(cohp_output_dir / "hist_cohp_B2avg_up.csv", index_col=0)
cohpB2avg_down = pd.read_csv(cohp_output_dir / "hist_cohp_B2avg_down.csv", index_col=0)
cohpB2avg_tot = pd.concat([cohpB2avg_up, cohpB2avg_down], ignore_index=True).T

normcohpB1avg_up = pd.read_csv(
    cohp_output_dir / "norm_hist_cohp_B1avg_up.csv", index_col=0
)
normcohpB1avg_down = pd.read_csv(
    cohp_output_dir / "norm_hist_cohp_B1avg_down.csv", index_col=0
)
normcohpB1avg_tot = pd.concat(
    [normcohpB1avg_up, normcohpB1avg_down], ignore_index=True
).T

normcohpB2avg_up = pd.read_csv(
    cohp_output_dir / "norm_hist_cohp_B2avg_up.csv", index_col=0
)
normcohpB2avg_down = pd.read_csv(
    cohp_output_dir / "norm_hist_cohp_B2avg_down.csv", index_col=0
)
normcohpB2avg_tot = pd.concat(
    [normcohpB2avg_up, normcohpB2avg_down], ignore_index=True
).T


# print(dosTtot)
def create_all_projections(
    data_dictionary: dict,
    metric_names: list,
    NN_list: list,
    min_dist: float = 0.1,
    connector_symbol: Literal["+", "*", "-"] = "*",
):
    """Creates UMAP projections for each key-val pair in data_dictionary, for all given distance metrics and nearest-neighbors values.
    The distance metrics must be given names recognized by UMAP.
    The metric 'wminkowski' (weighted minkowski) is hardcoded to turn into standardized manhattan (minkowski with p=1 and scaled by inverse std deviation)
    The data_dictionary keys will be used as projection name, the value must be a DataFrame or list of DataFrames.
    If the data is a list of DataFrames, it will project each DataFrame separately first and then combines the projections.

    Args:
        data_dictionary (dict[setname, DataFrame | list[DataFrame]]): collection of datasets to use for projection, values must be DataFrame or list of DataFrames.
                                                                        Currently only list lengths of 2 and 4 are supported
        metric_names (list): names of distance metrics to use. Must be in UMAPs list of metrics
        NN_list (list): list of nearest neighbor values to use in projection. All NN values will be used for each metric.
        min_dist (float, optional): value of the min_dist parameter to use for all projections. Varying min_dist is currently not supported. Defaults to 0.1.
        connector_symbol (Literal[, optional): CURRENTLY NOT SUPPORTED. Intended to change how separate projections are combined.
                                                Currently uses the intersection (*) for all. Defaults to '*'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    datnames = list(data_dictionary.keys())

    mcols = pd.MultiIndex.from_product([datnames, metric_names, NN_list, ["x", "y"]])
    # print(projection_names)

    proj_df = pd.DataFrame(index=data_dictionary[datnames[0]].index, columns=mcols)
    jj = 0
    for dataname, metric in product(datnames, metric_names):
        # print(data)
        print(f"precomputing k-NN for {metric}")

        if (
            type(data_dictionary[dataname]) == list
            and len(data_dictionary[dataname]) == 2
        ):
            if metric == "seuclidean":
                # calculate standard deviations to put in as weights for standardised euclidean
                # add 1e-4 to each std to compensate for 20 highest energy points being empty
                metric_kwd1 = {
                    "sigma": np.array(
                        data_dictionary[dataname][0].std(), dtype=np.float64
                    )
                    + np.ones(data_dictionary[dataname][0].shape[1]) * 0.0001
                }
                metric_kwd2 = {
                    "sigma": np.array(
                        data_dictionary[dataname][1].std(), dtype=np.float64
                    )
                    + np.ones(data_dictionary[dataname][1].shape[1]) * 0.0001
                }
                # print(metric_kwd['sigma'].shape)
            elif metric == "wminkowski":
                metric_kwd1 = {
                    "sigma": 1
                    / (
                        np.array(data_dictionary[dataname][0].std(), dtype=np.float64)
                        + np.ones(data_dictionary[dataname][0].shape[1]) * 0.0001
                    ),
                    "p": 1,
                }
                metric_kwd2 = {
                    "sigma": 1
                    / (
                        np.array(data_dictionary[dataname][1].std(), dtype=np.float64)
                        + np.ones(data_dictionary[dataname][1].shape[1]) * 0.0001
                    ),
                    "p": 1,
                }
            else:
                metric_kwd1 = {}
                metric_kwd2 = {}
            # precompute highest number of k-NN to speed up multiple projections
            knn1 = nearest_neighbors(
                data_dictionary[dataname][0],
                n_neighbors=max(NN_list),
                metric=metric,
                metric_kwds=metric_kwd1,
                angular=False,
                random_state=None,
            )
            knn2 = nearest_neighbors(
                data_dictionary[dataname][1],
                n_neighbors=max(NN_list),
                metric=metric,
                metric_kwds=metric_kwd2,
                angular=False,
                random_state=None,
            )
        elif (
            type(data_dictionary[dataname]) == list
            and len(data_dictionary[dataname]) == 4
        ):
            if metric == "seuclidean":
                # calculate standard deviations to put in as weights for standardised euclidean
                # add 1e-4 to each std to compensate for 20 highest energy points being empty
                metric_kwd1 = {
                    "sigma": np.array(
                        data_dictionary[dataname][0].std(), dtype=np.float64
                    )
                    + np.ones(data_dictionary[dataname][0].shape[1]) * 0.0001
                }
                metric_kwd2 = {
                    "sigma": np.array(
                        data_dictionary[dataname][1].std(), dtype=np.float64
                    )
                    + np.ones(data_dictionary[dataname][1].shape[1]) * 0.0001
                }
                metric_kwd3 = {
                    "sigma": np.array(
                        data_dictionary[dataname][2].std(), dtype=np.float64
                    )
                    + np.ones(data_dictionary[dataname][2].shape[1]) * 0.0001
                }
                metric_kwd4 = {
                    "sigma": np.array(
                        data_dictionary[dataname][3].std(), dtype=np.float64
                    )
                    + np.ones(data_dictionary[dataname][3].shape[1]) * 0.0001
                }
                # print(metric_kwd['sigma'].shape)
            elif metric == "wminkowski":
                metric_kwd1 = {
                    "sigma": 1
                    / (
                        np.array(data_dictionary[dataname][0].std(), dtype=np.float64)
                        + np.ones(data_dictionary[dataname][0].shape[1]) * 0.0001
                    ),
                    "p": 1,
                }
                metric_kwd2 = {
                    "sigma": 1
                    / (
                        np.array(data_dictionary[dataname][1].std(), dtype=np.float64)
                        + np.ones(data_dictionary[dataname][1].shape[1]) * 0.0001
                    ),
                    "p": 1,
                }
                metric_kwd3 = {
                    "sigma": 1
                    / (
                        np.array(data_dictionary[dataname][2].std(), dtype=np.float64)
                        + np.ones(data_dictionary[dataname][2].shape[1]) * 0.0001
                    ),
                    "p": 1,
                }
                metric_kwd4 = {
                    "sigma": 1
                    / (
                        np.array(data_dictionary[dataname][3].std(), dtype=np.float64)
                        + np.ones(data_dictionary[dataname][3].shape[1]) * 0.0001
                    ),
                    "p": 1,
                }
            else:
                metric_kwd1 = {}
                metric_kwd2 = {}
                metric_kwd3 = {}
                metric_kwd4 = {}
            # precompute highest number of k-NN to speed up multiple projections
            knn1 = nearest_neighbors(
                data_dictionary[dataname][0],
                n_neighbors=max(NN_list),
                metric=metric,
                metric_kwds=metric_kwd1,
                angular=False,
                random_state=None,
            )
            knn2 = nearest_neighbors(
                data_dictionary[dataname][1],
                n_neighbors=max(NN_list),
                metric=metric,
                metric_kwds=metric_kwd2,
                angular=False,
                random_state=None,
            )
            knn3 = nearest_neighbors(
                data_dictionary[dataname][2],
                n_neighbors=max(NN_list),
                metric=metric,
                metric_kwds=metric_kwd3,
                angular=False,
                random_state=None,
            )
            knn4 = nearest_neighbors(
                data_dictionary[dataname][3],
                n_neighbors=max(NN_list),
                metric=metric,
                metric_kwds=metric_kwd4,
                angular=False,
                random_state=None,
            )
        else:
            if metric == "seuclidean":
                # calculate standard deviations to put in as weights for standardised euclidean
                # add 1e-4 to each std to compensate for 20 highest energy points being empty
                metric_kwd = {
                    "sigma": np.array(data_dictionary[dataname].std(), dtype=np.float64)
                    + np.ones(data_dictionary[dataname].shape[1]) * 0.0001
                }
                print(metric_kwd["sigma"].shape)
            elif metric == "wminkowski":
                metric_kwd = {
                    "sigma": 1
                    / (
                        np.array(data_dictionary[dataname].std(), dtype=np.float64)
                        + np.ones(data_dictionary[dataname].shape[1]) * 0.0001
                    ),
                    "p": 1,
                }
            else:
                metric_kwd = {}
            knn = nearest_neighbors(
                data_dictionary[dataname],
                n_neighbors=max(NN_list),
                metric=metric,
                metric_kwds=metric_kwd,
                angular=False,
                random_state=None,
            )

        for nn_val in NN_list:

            print(f"working on projection: {dataname},{metric},{nn_val}")
            if (
                type(data_dictionary[dataname]) == list
                and len(data_dictionary[dataname]) == 2
            ):
                reducer1 = umap.UMAP(
                    n_neighbors=nn_val, min_dist=min_dist, precomputed_knn=knn1
                )
                reducer2 = umap.UMAP(
                    n_neighbors=nn_val, min_dist=min_dist, precomputed_knn=knn2
                )
                proj1 = reducer1.fit(data_dictionary[dataname][0])
                proj2 = reducer2.fit(data_dictionary[dataname][1])
                combined = proj1 * proj2
                proj_df[dataname, metric, nn_val, "x"] = combined.embedding_[:, 0]
                proj_df[dataname, metric, nn_val, "y"] = combined.embedding_[:, 1]
            elif (
                type(data_dictionary[dataname]) == list
                and len(data_dictionary[dataname]) == 4
            ):
                reducer1 = umap.UMAP(
                    n_neighbors=nn_val, min_dist=min_dist, precomputed_knn=knn1
                )
                reducer2 = umap.UMAP(
                    n_neighbors=nn_val, min_dist=min_dist, precomputed_knn=knn2
                )
                reducer3 = umap.UMAP(
                    n_neighbors=nn_val, min_dist=min_dist, precomputed_knn=knn3
                )
                reducer4 = umap.UMAP(
                    n_neighbors=nn_val, min_dist=min_dist, precomputed_knn=knn4
                )
                proj1 = reducer1.fit(data_dictionary[dataname][0])
                proj2 = reducer2.fit(data_dictionary[dataname][1])
                proj3 = reducer3.fit(data_dictionary[dataname][2])
                proj4 = reducer4.fit(data_dictionary[dataname][3])
                combined = proj1 * proj2 * proj3 * proj4
                proj_df[dataname, metric, nn_val, "x"] = combined.embedding_[:, 0]
                proj_df[dataname, metric, nn_val, "y"] = combined.embedding_[:, 1]

            elif type(data_dictionary[dataname]) == pd.DataFrame:
                reducer = umap.UMAP(
                    n_neighbors=nn_val, min_dist=min_dist, precomputed_knn=knn
                )
                proj = reducer.fit(data_dictionary[dataname])
                proj_df[dataname, metric, nn_val, "x"] = proj.embedding_[:, 0]
                proj_df[dataname, metric, nn_val, "y"] = proj.embedding_[:, 1]
            else:
                raise ValueError("Unimplemented amount of combined data")
            jj += 1

    return proj_df


if __name__ == "__main__":

    # Define datasets to use
    data_dict = {
        "TDOS [t_u,t_d]": dosTtot,
        "Norm. TDOS [t_u, t_d]": normdosTtot,
        "B.pDOS [B1_u,B1_d]*[B2_u,B2_d]": [dosB1tot, dosB2tot],
        "Norm. B.pDOS [B1_u,B1_d]*[B2_u,B2_d]": [normdosB1tot, normdosB2tot],
        "B.pDOS alt. [B1_u,B2_u]*[B1_d,B2_d]": [
            pd.concat([dosB1up, dosB2up], ignore_index=True).T,
            pd.concat([dosB1down, dosB2down], ignore_index=True).T,
        ],
        "Norm. B.pDOS alt. [B1_u,B2_u]*[B1_d,B2_d]": [
            pd.concat([normdosB1up, normdosB2up], ignore_index=True).T,
            pd.concat([normdosB1down, normdosB2down], ignore_index=True).T,
        ],
        "separated B.pDOS [B1_u]*[B1_d]*[B2_u]*[B2_d]": [
            dosB1up.T,
            dosB1down.T,
            dosB2up.T,
            dosB2down.T,
        ],
        "Norm. separated B.pDOS [B1_u]*[B1_d]*[B2_u]*[B2_d]": [
            normdosB1up.T,
            normdosB1down.T,
            normdosB2up.T,
            normdosB2down.T,
        ],
        "COHP avg(B-X) [B1_u,B1_d]*[B2_u,B2_d]": [cohpB1avg_tot, cohpB2avg_tot],
        "Norm.COHP avg(B-X) [B1_u,B1_d]*[B2_u,B2_d]": [
            normcohpB1avg_tot,
            normcohpB2avg_tot,
        ],
    }

    # trying numerical data from dcombined
    dcomb = pd.read_csv(
        "/home/lwalterb/hdp_project/umap_interactive/HDP_CombinedInfo_260418.csv",
        index_col=0,
    )

    # Metric and nearest neighbor list to use
    metric_list = [
        "euclidean",
        "seuclidean",
        "manhattan",
        "wminkowski",
        "braycurtis",
        "cosine",
    ]
    # metric_list = ['euclidean','manhattan','braycurtis']
    NN_list = [5, 15, 25, 50, 100]
    # NN_list = [5, 15, 25]

    proj_df = create_all_projections(
        data_dict,
        metric_names=metric_list,
        NN_list=NN_list,
        min_dist=0.1,
        connector_symbol="+",
    )
    print("writing csv")
    proj_df.to_csv(
        f"PrecomputedUMAPprojections_CombData{time.strftime('%y%m%d')}.csv",
        index=True,
        header=True,
    )
