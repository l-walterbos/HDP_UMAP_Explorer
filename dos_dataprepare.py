import pandas as pd
import numpy as np
import json
from monty.io import zopen
import os
from pathlib import Path


def get_histogrammed_curve(E_old, E_new, dos_data: np.ndarray, normalize: bool = False):
    """This function rebins a curve onto a new energy axis. It sums all DOS/COHP contributions of the old axis that
    fall within an energy bin in the new axis. This method has been adapted from the works:
    - Purcell et al., 2023, 10.1038/s41524-023-01063-y
    - Kuban et al., 2024, 10.1039/D4DD00258J

    One addition was made that if a certain bin is empty, it will take the average of the 2 neighboring values,
    but if dE is large enough compared to the original energy axis this will not be used.

    Args:
        E_old (np.ndarray): the old energy axis
        E_new (np.ndarray): the new energy axis to be histogrammed to
        dos_data (np.ndarray): the DOS or COHP data (or other general curve)
        normalize (bool, optional): whether to normalize the area of the histogrammed curve to equal 1. Defaults to False.

    Returns:
        np.ndarray: the curve rebinned to the new energy axis
    """
    dos_rebin = np.zeros(len(E_new) - 1)
    for ii, e1, e2 in zip(range(len(E_new) - 1), E_new[:-1], E_new[1:], strict=False):
        inds = np.where((E_old >= e1) & (E_old < e2))
        if ii > 0 and len(inds[0]) == 0 and dos_rebin[ii - 1] != 0:
            ind_low = np.where((E_old <= e1))[0]  # Closest value on negative side
            ind_high = np.where((E_old >= e2))[0]  # Closest value on positive side
            if len(ind_low) > 0 and len(ind_high) > 0:
                dos_rebin[ii] = (dos_data[ind_low[-1]] + dos_data[ind_high[0]]) / 2

        else:
            dos_rebin[ii] = np.sum(dos_data[inds])

    if normalize and np.sum(dos_rebin) != 0:
        return dos_rebin / (np.abs(np.sum(dos_rebin)) * (E_new[1] - E_new[0]))

    return dos_rebin


def interpolate_doscars(
    doscar_path: Path,
    dcomb: pd.DataFrame,
    ewindow: float | list[float] = 5.0,
    dE: float = 0.01,
    normalize: bool = False,
):
    """This function rebins the Density of States for the HDP compsitions in dcomb.
    The new energy axis will stretch -ewindow to max(bandgap)+ewindow with all VBM put to 0eV

    The function uses a histogramming method where DOS values inside the new energy interval are summed together
    If a new interval has no original points to sum over, it will take the average of the neighbouring values

    Args:
        doscar_path (Path): Path to the directory containing DOS jsons (XXXX_COMP_lsosmeareddos_per_site.json)
        dcomb (pd.DataFrame): Combined information DataFrame from CsBBX_Analyzer
        ewindow (float | list[float], optional): Energy window to include below VBM and above CBM (Can be single float or list of two floats) Defaults to 5.0.
        dE (float, optional): the energy interval for the new energy axis. Default to 0.01
        normalize (bool): whether to normalize the histogrammed



    Raises:
        ValueError: Could not parse the ewindow given

    Returns:
        Ecenters (np.ndarray): the centers of the new energy range
        dos(Tot/B1/B2/A/X)_(up/down) (dict[np.ndarray]): dictionaries containing the histogrammed (p)DOSs with the CompID as the keys

    """

    if type(ewindow) == float:

        # We shift each energy so VBM is at 0, so Emin = - Ewindow, Emax = max(bandgap) + Ewindow
        Emin = 0 - ewindow
        Emax = dcomb["bandgap"].max() + ewindow

        negative_window = ewindow
        positive_window = ewindow

    elif type(ewindow) == list and len(ewindow) == 2:

        Emin = 0 - abs(ewindow[0])
        Emax = dcomb["bandgap"].max() + ewindow[1]

        negative_window = abs(ewindow[0])
        positive_window = ewindow[1]
    else:
        raise ValueError(
            f"Parse Doscars got {ewindow} as parsing window, please provide a single float or list with len(2)"
        )

    dE = dE
    Enew = np.arange(Emin, Emax + dE, dE)

    dosA_up = {}
    dosA_down = {}
    dosB1_up = {}
    dosB1_down = {}
    dosB2_up = {}
    dosB2_down = {}
    dosX_up = {}
    dosX_down = {}
    dosTot_up = {}
    dosTot_down = {}

    # return Enew
    dcomb["element.B2"] = dcomb["element.B2"].fillna("Vac")

    ii = 0

    for comp in dcomb.index:
        dos_file = doscar_path / f"{comp}_lsosmeareddos_persite.json.gz"
        with zopen(dos_file, "rt") as f:
            dosdat = json.load(f)

        # Shift energies to make sure the VBM lays at 0eV
        energies_old = dosdat["tdos"]["energies"] - dcomb.loc[comp, "VBM"]
        # We want to include only the energies within Ewindow range from VBM and CBM
        include_index = [
            i
            for i in range(len(energies_old))
            if -negative_window
            <= energies_old[i]
            <= float(dcomb.loc[comp, "CBM"])
            - float(dcomb.loc[comp, "VBM"])
            + positive_window
        ]

        if dcomb.loc[comp, "element.B2"] == "Vac":
            site_map = {
                "A": ["1", "2"],
                "X": ["3", "4", "5", "6", "7", "8"],
                "B1": ["0"],
            }
            olddos_B1up = np.array(
                dosdat["tdos_per_site"][site_map["B1"][0]]["densities"]["1"]
            )[include_index]
            olddos_B1down = np.array(
                dosdat["tdos_per_site"][site_map["B1"][0]]["densities"]["-1"]
            )[include_index]

            olddos_B2up = np.zeros(len(include_index))
            olddos_B2down = np.zeros(len(include_index))
        else:
            site_map = {
                "A": ["2", "3"],
                "X": ["4", "5", "6", "7", "8", "9"],
                "B1": ["0"],
                "B2": ["1"],
            }
            olddos_B1up = np.array(
                dosdat["tdos_per_site"][site_map["B1"][0]]["densities"]["1"]
            )[include_index]
            olddos_B1down = np.array(
                dosdat["tdos_per_site"][site_map["B1"][0]]["densities"]["-1"]
            )[include_index]

            olddos_B2up = np.array(
                dosdat["tdos_per_site"][site_map["B2"][0]]["densities"]["1"]
            )[include_index]
            olddos_B2down = np.array(
                dosdat["tdos_per_site"][site_map["B2"][0]]["densities"]["-1"]
            )[include_index]

        olddos_Aup = np.zeros(len(include_index))
        olddos_Adown = np.zeros(len(include_index))

        for site_index in site_map["A"]:
            olddos_Aup += np.array(
                dosdat["tdos_per_site"][site_index]["densities"]["1"]
            )[include_index]
            olddos_Adown += np.array(
                dosdat["tdos_per_site"][site_index]["densities"]["-1"]
            )[include_index]

        olddos_Xup = np.zeros(len(include_index))
        olddos_Xdown = np.zeros(len(include_index))

        for site_index in site_map["X"]:
            olddos_Xup += np.array(
                dosdat["tdos_per_site"][site_index]["densities"]["1"]
            )[include_index]
            olddos_Xdown += np.array(
                dosdat["tdos_per_site"][site_index]["densities"]["-1"]
            )[include_index]

        olddos_Tup = np.array(dosdat["tdos"]["densities"]["1"])[include_index]
        olddos_Tdown = np.array(dosdat["tdos"]["densities"]["-1"])[include_index]
        Eold = energies_old[include_index]
        if np.float64(dcomb.loc[comp, "popdiff.total"]) >= 0.0:
            # flip the spins if magmom is aimed towards negative side
            dosA_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Aup, normalize=normalize
                    )
                }
            )
            dosA_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Adown, normalize=normalize
                    )
                }
            )

            dosB1_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_B1up, normalize=normalize
                    )
                }
            )
            dosB1_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_B1down, normalize=normalize
                    )
                }
            )

            dosB2_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_B2up, normalize=normalize
                    )
                }
            )
            dosB2_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_B2down, normalize=normalize
                    )
                }
            )

            dosX_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Xup, normalize=normalize
                    )
                }
            )
            dosX_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Xdown, normalize=normalize
                    )
                }
            )

            dosTot_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Tup, normalize=normalize
                    )
                }
            )
            dosTot_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Tdown, normalize=normalize
                    )
                }
            )
        else:
            dosA_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Adown, normalize=normalize
                    )
                }
            )
            dosA_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Aup, normalize=normalize
                    )
                }
            )

            dosB1_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_B1down, normalize=normalize
                    )
                }
            )
            dosB1_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_B1up, normalize=normalize
                    )
                }
            )

            dosB2_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_B2down, normalize=normalize
                    )
                }
            )
            dosB2_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_B2up, normalize=normalize
                    )
                }
            )

            dosX_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Xdown, normalize=normalize
                    )
                }
            )
            dosX_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Xup, normalize=normalize
                    )
                }
            )

            dosTot_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Tdown, normalize=normalize
                    )
                }
            )
            dosTot_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold, Enew, olddos_Tup, normalize=normalize
                    )
                }
            )

        ii += 1
        if ii % 100 == 0:
            print(f"working on comp {ii}")
    Ecenters = np.round(0.5 * (Enew[:-1] + Enew[1:]), 5)

    return (
        Ecenters,
        dosTot_up,
        dosTot_down,
        dosB1_up,
        dosB1_down,
        dosB2_up,
        dosB2_down,
        dosA_up,
        dosA_down,
        dosX_up,
        dosX_down,
    )


def interpolate_cohpcars(
    cohpcar_path: Path,
    dcomb: pd.DataFrame,
    ewindow: float | list[float] = 5.0,
    dE: float = 0.01,
    normalize: bool = False,
):
    """This function uses a binning method to equalize the energy axis for all COHPCARs in the data set.
    The VBM for all compositions will be set to 0. The new energy range will then span from VBM-ewindow to CBM + ewindow.

    The function uses a histogramming method where DOS values inside the new energy interval are summed together
    If a new interval has no original points to sum over, it will take the average of the neighbouring values

    Args:
        cohpcar_path (Path): Path to the directory containing DOS jsons (XXXX_COMP_smearedCOHP.json.gz)
        dcomb (pd.DataFrame): Combined information DataFrame from CsBBX_Analyzer
        ewindow (float | list[float], optional): Energy window to include below VBM and above CBM (Can be single float or list of two floats) Defaults to 5.0.
        dE (float, optional): the energy interval for the new energy axis. Default to 0.01
        normalize (bool): whether to normalize the histogrammed

    Raises:
        ValueError: The given ewindow could not be parsed

    Returns:
        Ecenters (np.ndarray): the centers of the new energy range
        cohp(B1/B2)_(avg/x/y/z)_(up/down) (dict[np.ndarray]): dictionaries containing the histogrammed (p)COHPs with the CompID as the keys
    """
    if type(ewindow) == float:

        # We shift each energy so VBM is at 0, so Emin = - Ewindow, Emax = max(bandgap) + Ewindow
        Emin = 0 - ewindow
        Emax = dcomb["bandgap"].max() + ewindow

        negative_window = ewindow
        positive_window = ewindow

    elif type(ewindow) == list and len(ewindow) == 2:

        Emin = 0 - abs(ewindow[0])
        Emax = dcomb["bandgap"].max() + ewindow[1]

        negative_window = abs(ewindow[0])
        positive_window = ewindow[1]
    else:
        raise ValueError(
            f"Parse Doscars got {ewindow} as parsing window, please provide a single float or list with len(2)"
        )

    dE = dE
    Enew = np.arange(Emin, Emax + dE, dE)

    cohpB1avg_up = {}
    cohpB1avg_down = {}
    cohpB1x_up = {}
    cohpB1x_down = {}
    cohpB1y_up = {}
    cohpB1y_down = {}
    cohpB1z_up = {}
    cohpB1z_down = {}

    cohpB2avg_up = {}
    cohpB2avg_down = {}
    cohpB2x_up = {}
    cohpB2x_down = {}
    cohpB2y_up = {}
    cohpB2y_down = {}
    cohpB2z_up = {}
    cohpB2z_down = {}

    # print(Enew)
    dcomb["element.B2"] = dcomb["element.B2"].fillna("Vac")

    ii = 0

    for comp in dcomb.index:
        dos_file = cohpcar_path / f"{comp}_smearedCOHP.json.gz"
        with zopen(dos_file, "rt") as f:
            cohpdat = json.load(f)

        # Shift energies to make sure the VBM lays at 0eV
        energies_old = cohpdat["energies"] - dcomb.loc[comp, "VBM"] - cohpdat["efermi"]
        # We want to include only the energies within Ewindow range from VBM and CBM
        include_index = [
            i
            for i in range(len(energies_old))
            if -negative_window
            <= energies_old[i]
            <= float(dcomb.loc[comp, "bandgap"]) + positive_window
        ]
        Eold = energies_old[include_index]
        # print(Eold)
        if np.float64(dcomb.loc[comp, "popdiff.total"]) >= 0.0:
            spinup = "1"
            spindown = "-1"
        else:
            spinup = "-1"
            spindown = "1"

        cohpB1avg_up.update(
            {
                comp: get_histogrammed_curve(
                    Eold,
                    Enew,
                    np.array(cohpdat["coxx_b1avg"][spinup])[include_index],
                    normalize=normalize,
                )
            }
        )
        cohpB1avg_down.update(
            {
                comp: get_histogrammed_curve(
                    Eold,
                    Enew,
                    np.array(cohpdat["coxx_b1avg"][spindown])[include_index],
                    normalize=normalize,
                )
            }
        )
        cohpB1x_up.update(
            {
                comp: get_histogrammed_curve(
                    Eold,
                    Enew,
                    np.array(cohpdat["coxx_b1x"][spinup])[include_index],
                    normalize=normalize,
                )
            }
        )
        cohpB1x_down.update(
            {
                comp: get_histogrammed_curve(
                    Eold,
                    Enew,
                    np.array(cohpdat["coxx_b1x"][spindown])[include_index],
                    normalize=normalize,
                )
            }
        )
        cohpB1y_up.update(
            {
                comp: get_histogrammed_curve(
                    Eold,
                    Enew,
                    np.array(cohpdat["coxx_b1y"][spinup])[include_index],
                    normalize=normalize,
                )
            }
        )
        cohpB1y_down.update(
            {
                comp: get_histogrammed_curve(
                    Eold,
                    Enew,
                    np.array(cohpdat["coxx_b1y"][spindown])[include_index],
                    normalize=normalize,
                )
            }
        )
        cohpB1z_up.update(
            {
                comp: get_histogrammed_curve(
                    Eold,
                    Enew,
                    np.array(cohpdat["coxx_b1z"][spinup])[include_index],
                    normalize=normalize,
                )
            }
        )
        cohpB1z_down.update(
            {
                comp: get_histogrammed_curve(
                    Eold,
                    Enew,
                    np.array(cohpdat["coxx_b1z"][spindown])[include_index],
                    normalize=normalize,
                )
            }
        )

        if dcomb.loc[comp, "element.B2"] == "Vac":
            cohpB2avg_up.update({comp: np.zeros(len(Enew) - 1)})
            cohpB2avg_down.update({comp: np.zeros(len(Enew) - 1)})
            cohpB2x_up.update({comp: np.zeros(len(Enew) - 1)})
            cohpB2x_down.update({comp: np.zeros(len(Enew) - 1)})
            cohpB2y_up.update({comp: np.zeros(len(Enew) - 1)})
            cohpB2y_down.update({comp: np.zeros(len(Enew) - 1)})
            cohpB2z_up.update({comp: np.zeros(len(Enew) - 1)})
            cohpB2z_down.update({comp: np.zeros(len(Enew) - 1)})
        else:
            cohpB2avg_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold,
                        Enew,
                        np.array(cohpdat["coxx_b2avg"][spinup])[include_index],
                        normalize=normalize,
                    )
                }
            )
            cohpB2avg_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold,
                        Enew,
                        np.array(cohpdat["coxx_b2avg"][spindown])[include_index],
                        normalize=normalize,
                    )
                }
            )
            cohpB2x_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold,
                        Enew,
                        np.array(cohpdat["coxx_b2x"][spinup])[include_index],
                        normalize=normalize,
                    )
                }
            )
            cohpB2x_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold,
                        Enew,
                        np.array(cohpdat["coxx_b2x"][spindown])[include_index],
                        normalize=normalize,
                    )
                }
            )
            cohpB2y_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold,
                        Enew,
                        np.array(cohpdat["coxx_b2y"][spinup])[include_index],
                        normalize=normalize,
                    )
                }
            )
            cohpB2y_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold,
                        Enew,
                        np.array(cohpdat["coxx_b2y"][spindown])[include_index],
                        normalize=normalize,
                    )
                }
            )
            cohpB2z_up.update(
                {
                    comp: get_histogrammed_curve(
                        Eold,
                        Enew,
                        np.array(cohpdat["coxx_b2z"][spinup])[include_index],
                        normalize=normalize,
                    )
                }
            )
            cohpB2z_down.update(
                {
                    comp: get_histogrammed_curve(
                        Eold,
                        Enew,
                        np.array(cohpdat["coxx_b2z"][spindown])[include_index],
                        normalize=normalize,
                    )
                }
            )

        ii += 1
        if ii % 100 == 0:
            print(f"working on comp {ii}")
    Ecenters = np.round(0.5 * (Enew[:-1] + Enew[1:]), 5)

    return (
        Ecenters,
        cohpB1avg_up,
        cohpB1avg_down,
        cohpB1x_up,
        cohpB1x_down,
        cohpB1y_up,
        cohpB1y_down,
        cohpB1z_up,
        cohpB1z_down,
        cohpB2avg_up,
        cohpB2avg_down,
        cohpB2x_up,
        cohpB2x_down,
        cohpB2y_up,
        cohpB2y_down,
        cohpB2z_up,
        cohpB2z_down,
    )


def compare_plots(
    dcomb_path: Path,
    hist_dos_path: Path = Path("./histogrammed_doss"),
    hist_cohp_path: Path = Path("./histogrammed_cohps"),
    n_samples: int = 10,
    output_dir: Path = Path("."),
):
    """Samples n random compositions. Plots the DOS and COHP using the HDP_PlotslyPlots scripts, and adds curves from the histogrammed DOS and COHPs.
    With the histogramming method the amplitude at each point will be higher, but is used whether the overall shape has been perserved.

    Args:
        dcomb_path (Path): Path to saved CombinedInfo DataFrame
        hist_dos_path (Path, optional): Path to histogrammed DOS data. Defaults to Path('./histogrammed_doss').
        hist_cohp_path (Path, optional): Path to histogrammed COHP data. Defaults to Path('./histogrammed_cohps').
        n_samples (int, optional): Number of compositions to sample. Defaults to 10.
        output_dir (Path, optional): Where to save figures. Defaults to Path('.').
    """
    from HDP_PlotslyPlots import plot_coxx, plot_dos
    import plotly.graph_objects as go

    dinfo = pd.read_csv(dcomb_path, index_col=0)

    dtdos_up = pd.read_csv(hist_dos_path / "hist_tdos_up.csv", index_col=0)
    dtdos_down = pd.read_csv(hist_dos_path / "hist_tdos_down.csv", index_col=0)

    dcohp_b1up = pd.read_csv(hist_cohp_path / "hist_cohp_B1avg_up.csv", index_col=0)
    dcohp_b1down = pd.read_csv(hist_cohp_path / "hist_cohp_B1avg_down.csv", index_col=0)

    dcohp_b2up = pd.read_csv(hist_cohp_path / "hist_cohp_B2avg_up.csv", index_col=0)
    dcohp_b2down = pd.read_csv(hist_cohp_path / "hist_cohp_B2avg_down.csv", index_col=0)

    samples = list(dinfo.sample(n_samples).index)

    for ii in range(len(samples)):
        comp = samples[ii]

        dos_fig = plot_dos(
            comp,
            dcomb_path,
            dos_path=dosplot_pointer["Path"],
            dos_extension=dosplot_pointer["extension"],
        )
        dos_fig.add_trace(
            go.Scatter(x=np.array(dtdos_up[comp]), y=dtdos_up[comp].index)
        )
        dos_fig.add_trace(
            go.Scatter(x=-1 * np.array(dtdos_down[comp]), y=dtdos_down[comp].index)
        )
        dos_fig.write_html(output_dir / f"{ii}_{comp}_dos.html")

        cohp_figs = plot_coxx(
            comp,
            dcomb_path,
            "cohp",
            coxx_path=coxxplot_pointer["COHP"]["Path"],
            coxx_extenstion=coxxplot_pointer["COHP"]["extension"],
        )

        cohp_figs[0].add_trace(
            go.Scatter(x=-1 * np.array(dcohp_b1up[comp]), y=dcohp_b1up[comp].index)
        )
        cohp_figs[0].add_trace(
            go.Scatter(x=-1 * np.array(dcohp_b1down[comp]), y=dcohp_b1down[comp].index)
        )
        cohp_figs[0].write_html(output_dir / f"{ii}_{comp}_cohpb1.html")

        cohp_figs[1].add_trace(
            go.Scatter(x=-1 * np.array(dcohp_b2up[comp]), y=dcohp_b2up[comp].index)
        )
        cohp_figs[1].add_trace(
            go.Scatter(x=-1 * np.array(dcohp_b2down[comp]), y=dcohp_b2down[comp].index)
        )
        cohp_figs[1].write_html(output_dir / f"{ii}_{comp}_cohpb2.html")

    pass


dosplot_pointer = {
    "Path": Path("./lsodos_smeared"),
    "extension": "lsosmeareddos_persite.json.gz",
}
coxxplot_pointer = {
    "COHP": {"Path": Path("./cohps_smeared"), "extension": "smearedCOHP.json.gz"},
    "COBI": {"Path": Path("./cobis_smeared"), "extension": "smearedCOBI.json.gz"},
}


if __name__ == "__main__":

    info_df_path = Path("./HDP_CombinedInfo_260418.csv")
    dcomb = pd.read_csv(info_df_path, index_col=0)

    dos_path = Path("./lsodos_smeared/")
    cohp_path = Path("./cohps_smeared")
    dos_output_dir = Path("./smeared_histogrammed_doss")
    dos_output_dir.mkdir(exist_ok=True)
    dE = 0.1
    print("Doing standard DOS histograms")
    (
        Ecenters,
        dosTot_up,
        dosTot_down,
        dosB1_up,
        dosB1_down,
        dosB2_up,
        dosB2_down,
        dosA_up,
        dosA_down,
        dosX_up,
        dosX_down,
    ) = interpolate_doscars(dos_path, dcomb, dE=dE, normalize=False)
    histdos_Tup = (
        pd.DataFrame(index=Ecenters, data=dosTot_up)
        .round(5)
        .to_csv(dos_output_dir / "hist_tdos_up.csv")
    )
    histdos_Tdown = (
        pd.DataFrame(index=Ecenters, data=dosTot_down)
        .round(5)
        .to_csv(dos_output_dir / "hist_tdos_down.csv")
    )

    histdos_B1up = (
        pd.DataFrame(index=Ecenters, data=dosB1_up)
        .round(5)
        .to_csv(dos_output_dir / "hist_B1dos_up.csv")
    )
    histdos_B1down = (
        pd.DataFrame(index=Ecenters, data=dosB1_down)
        .round(5)
        .to_csv(dos_output_dir / "hist_B1dos_down.csv")
    )

    histdos_B2up = (
        pd.DataFrame(index=Ecenters, data=dosB2_up)
        .round(5)
        .to_csv(dos_output_dir / "hist_B2dos_up.csv")
    )
    histdos_B2down = (
        pd.DataFrame(index=Ecenters, data=dosB2_down)
        .round(5)
        .to_csv(dos_output_dir / "hist_B2dos_down.csv")
    )

    histdos_Aup = (
        pd.DataFrame(index=Ecenters, data=dosA_up)
        .round(5)
        .to_csv(dos_output_dir / "hist_Ados_up.csv")
    )
    histdos_Adown = (
        pd.DataFrame(index=Ecenters, data=dosA_down)
        .round(5)
        .to_csv(dos_output_dir / "hist_Ados_down.csv")
    )

    histdos_Xup = (
        pd.DataFrame(index=Ecenters, data=dosX_up)
        .round(5)
        .to_csv(dos_output_dir / "hist_Xdos_up.csv")
    )
    histdos_Xdown = (
        pd.DataFrame(index=Ecenters, data=dosX_down)
        .round(5)
        .to_csv(dos_output_dir / "hist_Xdos_down.csv")
    )

    print("Doing normalized DOS histograms")
    (
        Ecenters,
        dosTot_up,
        dosTot_down,
        dosB1_up,
        dosB1_down,
        dosB2_up,
        dosB2_down,
        dosA_up,
        dosA_down,
        dosX_up,
        dosX_down,
    ) = interpolate_doscars(dos_path, dcomb, dE=dE, normalize=True)
    histdos_Tup = (
        pd.DataFrame(index=Ecenters, data=dosTot_up)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_tdos_up.csv")
    )
    histdos_Tdown = (
        pd.DataFrame(index=Ecenters, data=dosTot_down)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_tdos_down.csv")
    )

    histdos_B1up = (
        pd.DataFrame(index=Ecenters, data=dosB1_up)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_B1dos_up.csv")
    )
    histdos_B1down = (
        pd.DataFrame(index=Ecenters, data=dosB1_down)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_B1dos_down.csv")
    )

    histdos_B2up = (
        pd.DataFrame(index=Ecenters, data=dosB2_up)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_B2dos_up.csv")
    )
    histdos_B2down = (
        pd.DataFrame(index=Ecenters, data=dosB2_down)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_B2dos_down.csv")
    )

    histdos_Aup = (
        pd.DataFrame(index=Ecenters, data=dosA_up)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_Ados_up.csv")
    )
    histdos_Adown = (
        pd.DataFrame(index=Ecenters, data=dosA_down)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_Ados_down.csv")
    )

    histdos_Xup = (
        pd.DataFrame(index=Ecenters, data=dosX_up)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_Xdos_up.csv")
    )
    histdos_Xdown = (
        pd.DataFrame(index=Ecenters, data=dosX_down)
        .round(5)
        .to_csv(dos_output_dir / "norm_hist_Xdos_down.csv")
    )

    cohp_output_dir = Path("./smeared_histogrammed_cohps")
    cohp_output_dir.mkdir(exist_ok=True)
    print("Doing standard COHP histograms")

    (
        Ecenters,
        cohpB1avg_up,
        cohpB1avg_down,
        cohpB1x_up,
        cohpB1x_down,
        cohpB1y_up,
        cohpB1y_down,
        cohpB1z_up,
        cohpB1z_down,
        cohpB2avg_up,
        cohpB2avg_down,
        cohpB2x_up,
        cohpB2x_down,
        cohpB2y_up,
        cohpB2y_down,
        cohpB2z_up,
        cohpB2z_down,
    ) = interpolate_cohpcars(cohp_path, dcomb, dE=dE, normalize=False)

    histcohp_b1avg_up = (
        pd.DataFrame(index=Ecenters, data=cohpB1avg_up)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B1avg_up.csv")
    )
    histcohp_b1avg_down = (
        pd.DataFrame(index=Ecenters, data=cohpB1avg_down)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B1avg_down.csv")
    )

    histcohp_b1x_up = (
        pd.DataFrame(index=Ecenters, data=cohpB1x_up)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B1x_up.csv")
    )
    histcohp_b1x_down = (
        pd.DataFrame(index=Ecenters, data=cohpB1x_down)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B1x_down.csv")
    )

    histcohp_b1y_up = (
        pd.DataFrame(index=Ecenters, data=cohpB1y_up)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B1y_up.csv")
    )
    histcohp_b1y_down = (
        pd.DataFrame(index=Ecenters, data=cohpB1y_down)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B1y_down.csv")
    )

    histcohp_b1z_up = (
        pd.DataFrame(index=Ecenters, data=cohpB1z_up)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B1z_up.csv")
    )
    histcohp_b1z_down = (
        pd.DataFrame(index=Ecenters, data=cohpB1z_down)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B1z_down.csv")
    )

    histcohp_b2avg_up = (
        pd.DataFrame(index=Ecenters, data=cohpB2avg_up)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B2avg_up.csv")
    )
    histcohp_b2avg_down = (
        pd.DataFrame(index=Ecenters, data=cohpB2avg_down)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B2avg_down.csv")
    )

    histcohp_b2x_up = (
        pd.DataFrame(index=Ecenters, data=cohpB2x_up)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B2x_up.csv")
    )
    histcohp_b2x_down = (
        pd.DataFrame(index=Ecenters, data=cohpB2x_down)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B2x_down.csv")
    )

    histcohp_b2y_up = (
        pd.DataFrame(index=Ecenters, data=cohpB2y_up)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B2y_up.csv")
    )
    histcohp_b2y_down = (
        pd.DataFrame(index=Ecenters, data=cohpB2y_down)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B2y_down.csv")
    )

    histcohp_b2z_up = (
        pd.DataFrame(index=Ecenters, data=cohpB2z_up)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B2z_up.csv")
    )
    histcohp_b2z_down = (
        pd.DataFrame(index=Ecenters, data=cohpB2z_down)
        .round(5)
        .to_csv(cohp_output_dir / "hist_cohp_B2z_down.csv")
    )

    print("Doing normalized COHP histograms")
    (
        Ecenters,
        cohpB1avg_up,
        cohpB1avg_down,
        cohpB1x_up,
        cohpB1x_down,
        cohpB1y_up,
        cohpB1y_down,
        cohpB1z_up,
        cohpB1z_down,
        cohpB2avg_up,
        cohpB2avg_down,
        cohpB2x_up,
        cohpB2x_down,
        cohpB2y_up,
        cohpB2y_down,
        cohpB2z_up,
        cohpB2z_down,
    ) = interpolate_cohpcars(cohp_path, dcomb, dE=dE, normalize=True)

    histcohp_b1avg_up = (
        pd.DataFrame(index=Ecenters, data=cohpB1avg_up)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B1avg_up.csv")
    )
    histcohp_b1avg_down = (
        pd.DataFrame(index=Ecenters, data=cohpB1avg_down)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B1avg_down.csv")
    )

    histcohp_b1x_up = (
        pd.DataFrame(index=Ecenters, data=cohpB1x_up)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B1x_up.csv")
    )
    histcohp_b1x_down = (
        pd.DataFrame(index=Ecenters, data=cohpB1x_down)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B1x_down.csv")
    )

    histcohp_b1y_up = (
        pd.DataFrame(index=Ecenters, data=cohpB1y_up)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B1y_up.csv")
    )
    histcohp_b1y_down = (
        pd.DataFrame(index=Ecenters, data=cohpB1y_down)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B1y_down.csv")
    )

    histcohp_b1z_up = (
        pd.DataFrame(index=Ecenters, data=cohpB1z_up)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B1z_up.csv")
    )
    histcohp_b1z_down = (
        pd.DataFrame(index=Ecenters, data=cohpB1z_down)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B1z_down.csv")
    )

    histcohp_b2avg_up = (
        pd.DataFrame(index=Ecenters, data=cohpB2avg_up)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B2avg_up.csv")
    )
    histcohp_b2avg_down = (
        pd.DataFrame(index=Ecenters, data=cohpB2avg_down)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B2avg_down.csv")
    )

    histcohp_b2x_up = (
        pd.DataFrame(index=Ecenters, data=cohpB2x_up)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B2x_up.csv")
    )
    histcohp_b2x_down = (
        pd.DataFrame(index=Ecenters, data=cohpB2x_down)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B2x_down.csv")
    )

    histcohp_b2y_up = (
        pd.DataFrame(index=Ecenters, data=cohpB2y_up)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B2y_up.csv")
    )
    histcohp_b2y_down = (
        pd.DataFrame(index=Ecenters, data=cohpB2y_down)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B2y_down.csv")
    )

    histcohp_b2z_up = (
        pd.DataFrame(index=Ecenters, data=cohpB2z_up)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B2z_up.csv")
    )
    histcohp_b2z_down = (
        pd.DataFrame(index=Ecenters, data=cohpB2z_down)
        .round(5)
        .to_csv(cohp_output_dir / "norm_hist_cohp_B2z_down.csv")
    )

    output_dir = Path("./histogram_smeared_test_plots")
    output_dir.mkdir(exist_ok=True)

    compare_plots(
        info_df_path,
        hist_dos_path=dos_output_dir,
        hist_cohp_path=cohp_output_dir,
        n_samples=20,
        output_dir=output_dir,
    )
