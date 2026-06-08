from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from HDP_PlotslyPlots import plot_dos, plot_coxx, plot_ptable
import umap
from umap.umap_ import nearest_neighbors
import warnings

warnings.filterwarnings("ignore")


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__)

##import data
info_df_path = Path("./HDP_CombinedInfo_260418.csv")
dinfo = pd.read_csv(info_df_path, index_col=0)
###Assure that majority spin == spin-up
dinfo["popdiff.B1"] = dinfo.apply(
    lambda row: (
        -row["popdiff.B1"] if row["popdiff.total"] <= 0.0 else row["popdiff.B1"]
    ),
    axis=1,
)
dinfo["popdiff.B2"] = dinfo.apply(
    lambda row: (
        -row["popdiff.B2"] if row["popdiff.total"] <= 0.0 else row["popdiff.B2"]
    ),
    axis=1,
)
dinfo["popdiff.total"] = abs(dinfo["popdiff.total"])
dinfo["element.B2"] = dinfo["element.B2"].fillna("Vac")

###Separate values to be able to use as color scale in UMAP plot
dplot = dinfo[
    [
        "comp_name_full",
        "bandgap",
        "element.X",
        "popdiff.total",
        "cond_type",
        "charge.B1",
        "charge.B2",
        "charge.X",
        "charge.A",
        "block.B1",
        "block.B2",
        "block_pairing",
        "transition_sites",
        "transition_bands",
        "spin_forbidden",
        "lattice_a_primitive",
        "size_Oh_B1",
        "size_Oh_B2",
    ]
]
###Introduce some additional columns, combining B1 and B2 values in several ways
dplot["antiparallel_magmom"] = [
    np.sign(x) != np.sign(y) if abs(x) > 0.1 and abs(y) > 0.1 else False
    for x, y in zip(dinfo["popdiff.B1"].values, dinfo["popdiff.B2"].fillna(0).values)
]
dplot["Icohp.sum"] = dinfo["Icohp.B1.sum"] + dinfo["Icohp.B2.sum"].fillna(0)
dplot["Icohp.diff"] = np.abs(dinfo["Icohp.B1.sum"] - dinfo["Icohp.B2.sum"].fillna(0))
dplot["Icobi.sum"] = dinfo["Icobi.B1.sum"] + dinfo["Icobi.B2.sum"].fillna(0)
dplot["Icobi.diff"] = np.abs(dinfo["Icobi.B1.sum"] - dinfo["Icobi.B2.sum"].fillna(0))
dplot["charge.diff"] = np.abs(dinfo["charge.B1"] - dinfo["charge.B2"].fillna(0))
dplot["Icohp.total_dir_asym"] = dinfo["Icohp.B1.directional_asym_index"] + dinfo[
    "Icohp.B2.directional_asym_index"
].fillna(0)
dplot["Icobi.total_dir_asym"] = dinfo["Icobi.B1.directional_asym_index"] + dinfo[
    "Icobi.B2.directional_asym_index"
].fillna(0)
dplot["Icohp.total_axial_asym"] = dinfo["Icohp.B1.axial_asym_index"] + dinfo[
    "Icohp.B2.axial_asym_index"
].fillna(0)
dplot["Icobi.total_axial_asym"] = dinfo["Icobi.B1.axial_asym_index"] + dinfo[
    "Icobi.B2.axial_asym_index"
].fillna(0)
dplot["Octh. Ratio"] = dplot.apply(
    lambda row: (
        row["size_Oh_B1"] / row["size_Oh_B2"]
        if row["size_Oh_B2"] > row["size_Oh_B1"]
        else row["size_Oh_B2"] / row["size_Oh_B1"]
    ),
    axis=1,
)
# dinfo.apply(lambda row:  if (abs(row['popdiff.B1']) > 0.1 and abs(row['popdiff.B2']) >0.1 ) np.sign(row['popdiff.B1'] != np.sign(row['popdiff.B2']) else False, axis=1))
#    np.sign(dinfo['popdiff.B1']) != np.sign(dinfo['popdiff.B2'].fillna(0))) and ((abs(dinfo['popdiff.B1']) >0.1) and (abs(dinfo['popdiff.B2']) >0.1))
used_nn = 0

# path to preomputed umap projections.
precomp_umap_path = Path("./PrecomputedUMAPprojections_smeared260419.csv")
dumap = pd.read_csv(precomp_umap_path, index_col=0, header=[0, 1, 2, 3])

# available values to color the periodic table plot by.
ptable_color_list = [
    "Element Counts",
    "Average -ICOHP",
    "Average ICOBI",
    "Average Dir.Asym.Index (ICOHP)",
    "Average Axial.Asym.Index (ICOHP)",
    "Average Dir.Asym.Index (ICOBI)",
    "Average Axial.Asym.Index (ICOBI)",
    "Average popdiff",
    "Average Band Gap (eV)",
    "Average B-charge",
    "Average X-charge",
]


###Pointers used to point to DOS and COHP/COBI files
dosplot_pointer = {
    "Path": Path("./lsodos_smeared"),
    "extension": "lsosmeareddos_persite.json.gz",
}
coxxplot_pointer = {
    "COHP": {"Path": Path("./cohps_smeared"), "extension": "smearedCOHP.json.gz"},
    "COBI": {"Path": Path("./cobis_smeared"), "extension": "smearedCOBI.json.gz"},
}

# umapfig.update_traces(customdata=dplot.index)


app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H2("UMAP projection settings"),
                        html.Dt("Select UMAP projection data:"),
                        dcc.Dropdown(
                            list(dumap.columns.get_level_values(0).unique()),
                            value=list(dumap.columns.get_level_values(0).unique())[0],
                            id="umap-proj-data-dropdown",
                            style={"width": "70%"},
                        ),
                        html.Dt("Select metric:"),
                        dcc.RadioItems(
                            list(dumap.columns.get_level_values(1).unique()),
                            value=list(dumap.columns.get_level_values(1).unique())[-1],
                            id="umap-proj-metric-radio",
                            inline=True,
                        ),
                        html.Dt("Select NN value (min_dist=0.1)"),
                        html.Div(
                            [
                                dcc.Slider(
                                    min=0,
                                    max=list(
                                        dumap.columns.get_level_values(2)
                                        .astype(int)
                                        .unique()
                                    )[-1],
                                    step=None,
                                    marks={
                                        n: n
                                        for n in list(
                                            dumap.columns.get_level_values(2).unique()
                                        )
                                    },
                                    value=list(
                                        dumap.columns.get_level_values(2).astype(int)
                                    )[0],
                                    id="umap-proj-nn-slider",
                                )
                            ],
                            style={"display": "inline-block", "width": "80%"},
                        ),
                    ],
                    style={"width": "40%", "display": "inline-block"},
                ),
                html.Div([
                            html.H2("Overwrite UMAP colorscale range"),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            dcc.Input(
                                                type="number",
                                                placeholder="minimal val.",
                                                step=0.001,
                                                style={"width": "90%"},
                                                id="range_scale_min_input",
                                            )
                                        ],
                                        style={
                                            "width": "50%",
                                            # "display": "inline-block",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            dcc.Input(
                                                type="number",
                                                placeholder="maximal val.",
                                                step=0.001,
                                                style={"width": "90%"},
                                                id="range_scale_max_input",
                                            )
                                        ],
                                        style={
                                            "width": "50%",
                                            # "display": "inline-block",
                                        },
                                    ),
                                ]
                            ),
                        ],
                        style={"width": "30%","display": "inline-block"} 
                ),
                html.Div(
                    [
                        html.H2("Data filtering"),
                        html.Dt("Filter: comps containing B-sites:"),
                        dcc.Input(
                            type="text",
                            id="bsite_filter_input",
                            placeholder='Select B-sites, e.g., "Ag,Bi,Au"...',
                        ),
                        html.Dt("Filter: restrict B-sites to:"),
                        dcc.Input(
                            type="text",
                            id="bsite_restricter_input",
                            placeholder='Select B-sites, e.g., "Ag,Bi,Au"...',
                        ),
                        html.Dt("Select X-sites:"),
                        dcc.Checklist(
                            ["F", "Cl", "Br", "I"],
                            ["F", "Cl", "Br", "I"],
                            id="xsite_checklist",
                            inline=True,
                        ),
                    ],
                    style={"width": "30%", "display": "inline-block"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="UMAP-plot",
                            responsive=True,
                            # figure=umapfig,
                            clickData={"points": [{"hovertext": "1000_CsAgLuCl"}]},
                        ),
                    ],
                    style={
                        "width": "50%",
                        "height": "650px",
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Dt("UMAP Color Options:"),
                                dcc.Dropdown(
                                    options=dplot.columns.to_list(),
                                    value="bandgap",
                                    id="color-selector",
                                    style={"width": "80%"},
                                ),
                                dcc.RadioItems(
                                    ["Linear", "Log"],
                                    "Linear",
                                    id="color-scaling-radio",
                                    inline=True,
                                ),
                            ],
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.Dt("Ptable Color Options:", id="ptable-output"),
                                dcc.Dropdown(
                                    ptable_color_list,
                                    value=ptable_color_list[0],
                                    id="ptable_color_dropdown",
                                    style={"width": "80%"},
                                ),
                                dcc.RadioItems(
                                    ["Linear", "Log"],
                                    "Linear",
                                    id="ptable_scaling_radio",
                                    inline=True,
                                ),
                            ],
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        dcc.Graph(
                            id="ptable-plot", responsive=True, style={"height": "500px"}
                        ),
                        html.Dd("____"),
                    ],
                    style={
                        "width": "45%",
                        "display": "inline-block",
                        "height": "650px",
                    },
                ),
            ],
            style={"width": "100%"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Dt("Select Bonding Plot:"),
                        dcc.Dropdown(
                            ["COHP", "COBI"], value="COHP", id="coxx_selector"
                        ),
                        html.Dd("_____"),
                        html.Dt(id="disc-comp-name"),
                        html.Dd(id="disc-bandgap"),
                        html.Dd(id="disc-totalspin"),
                        html.Dd(id="disc-b1spin"),
                        html.Dd(id="disc-b2spin"),
                        html.Dt("Loewdin Charges"),
                        html.Dd(id="disc-chargeA"),
                        html.Dd(id="disc-chargeB1"),
                        html.Dd(id="disc-chargeB2"),
                        html.Dd(id="disc-chargeX"),
                        html.Dt("Bonding Data"),
                        html.Dd(id="disc-icohpB1"),
                        html.Dd(id="disc-icohpB2"),
                        html.Dd(id="disc-icobiB1"),
                        html.Dd(id="disc-icobiB2"),
                    ],
                    style={"display": "inline-block", "width": "10%"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="DOS-plot", style=dict(height="600px")),
                    ],
                    style={"display": "inline-block", "width": "30%"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="COXX1-plot", style=dict(height="600px")),
                    ],
                    style={"display": "inline-block", "width": "30%"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="COXX2-plot", style=dict(height="600px")),
                    ],
                    style={"display": "inline-block", "width": "30%"},
                ),
            ],
        ),
        # html.Div(dcc.Slider(
        #     df['Year'].min(),
        #     df['Year'].max(),
        #     step=None,
        #     id='crossfilter-year--slider',
        #     value=df['Year'].max(),
        #     marks={str(year): str(year) for year in df['Year'].unique()}
        # ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
    ]
)


@callback(
    Output("DOS-plot", "figure"),
    Output("COXX1-plot", "figure"),
    Output("COXX2-plot", "figure"),
    Input("UMAP-plot", "clickData"),
    Input("coxx_selector", "value"),
)
def update_sidegraphs(selected_point, selected_coxx):
    """When a new composition is clicked in the UMAP figure, this function updates the DOS and COHP plots

    Args:
        selected_point (str): CompID of clicked composition
        selected_coxx (str): from dropdown whether to plot COHP or COBI plots

    Returns:
        go.Figure: DOS, COHP[B1], COHP[B2] plots
    """
    selected_comp = selected_point["points"][0]["hovertext"]

    dosfig = plot_dos(
        selected_comp,
        info_df_path,
        dos_path=dosplot_pointer["Path"],
        dos_extension=dosplot_pointer["extension"],
    )
    coxxfigs = plot_coxx(
        selected_comp,
        info_df_path,
        coxx_type=selected_coxx.lower(),
        coxx_path=coxxplot_pointer[selected_coxx.upper()]["Path"],
        coxx_extenstion=coxxplot_pointer[selected_coxx.upper()]["extension"],
    )
    return dosfig, coxxfigs[0], coxxfigs[1]


@callback(
    Output("ptable-plot", "figure"),
    Input("ptable_color_dropdown", "value"),
    Input("ptable_scaling_radio", "value"),
)
def update_ptable_color(selected_color_col, selected_scale):
    """Used to update the periodic table plot if heatmap value or lin/log scale value has been changed

    Args:
        selected_color_col (str): which datacolumn to use for coloring element boxes
        selected_scale (str[lin|log]): whether to scale linearly of logarithmicly

    Returns:
        go.Figure: periodic table plot displaying selected average value for the elements
    """
    uselog = selected_scale == "Log"
    fig = plot_ptable(dinfo, color_value=selected_color_col, use_log=uselog)
    return fig


# @callback(
#         Output('ptableoutput', 'children'),
#         Input('ptable-plot','clickData'))
# def show_clicdata(clickat):
#     return f'{clickat}'


@callback(
    Output("disc-comp-name", "children"),
    Output("disc-bandgap", "children"),
    Output("disc-totalspin", "children"),
    Output("disc-b1spin", "children"),
    Output("disc-b2spin", "children"),
    Output("disc-chargeA", "children"),
    Output("disc-chargeB1", "children"),
    Output("disc-chargeB2", "children"),
    Output("disc-chargeX", "children"),
    Output("disc-icohpB1", "children"),
    Output("disc-icohpB2", "children"),
    Output("disc-icobiB1", "children"),
    Output("disc-icobiB2", "children"),
    Input("UMAP-plot", "clickData"),
)
def update_compdiscription(selected_point):
    """If a new point is clicked in the UMAP plot, this function updates the summary of composition data

    Args:
        selected_point (str): CompID of selected composition. Used to index CombinedInfo DataFrame

    Returns:
        str: several strings containing the data
    """
    selected_comp = selected_point["points"][0]["hovertext"]
    compseries = dinfo.loc[selected_comp]
    namestring = (
        f"ID#: {compseries['compID_num']}, Comp: {compseries['comp_name_full']}"
    )
    elecdatastring = [
        f"Bandgap: {compseries['bandgap']:.2f} eV",
        f"Net.Spin: {compseries['popdiff.total']}",
        f"Net.Spin[B1]: {compseries['popdiff.B1']}",
        f"Net.Spin[B2]: {compseries['popdiff.B2']}",
    ]
    chargestrings = [
        f"Charge A: {compseries['charge.A']}",
        f"Charge B1: {compseries['charge.B1']}",
        f"Charge B2: {compseries['charge.B2']}",
        f"Charge X: {round(compseries['charge.X'],2)}",
    ]
    coxxstrings = [
        f"ICOHP B1-X: {compseries['Icohp.B1.avg']:.2f} eV",
        f"ICOHP B2-X: {compseries['Icohp.B2.avg']:.2f} eV",
        f"ICOBI B1-X: {compseries['Icobi.B1.avg']:.2f}",
        f"ICOBI B2-X: {compseries['Icobi.B2.avg']:.2f}",
    ]
    return namestring, *elecdatastring, *chargestrings, *coxxstrings


@callback(
    Output("UMAP-plot", "figure"),
    Input("color-selector", "value"),
    Input("color-scaling-radio", "value"),
    Input("umap-proj-data-dropdown", "value"),
    Input("umap-proj-metric-radio", "value"),
    Input("umap-proj-nn-slider", "value"),
    Input("bsite_filter_input", "value"),
    Input("bsite_restricter_input", "value"),
    Input("xsite_checklist", "value"),
    Input("range_scale_min_input", "value"),
    Input("range_scale_max_input", "value"),
)
def change_color_umap(
    selected_column,
    color_scale,
    selected_data,
    selected_metric,
    nn_value,
    bsite_filter,
    bsite_restricter,
    xsite_filter,
    min_range_val,
    max_range_val,
):
    """Updates the color scale of UMAP plot

    Args:
        selected_column (str): which column of dplot DataFrame to use for coloring
        color_scale (str[lin|log]): whether to apply log scaling
        selected_data (str): Dataset name used for current umap projection
        selected_metric (str): which metric is currenlty being displayed
        nn_value (str): the current NN value from UMAP projection
        bsite_filter (list[str]): Data from B-site filter, filters all compositions that contain that ion on a B-site
        bsite_restricter (list[str]): Data from B-site restricter. Filters composition restricting ions to specified ions
        xsite_filter (list[str]): Checklist allowing specific halide compositions to be displayed
        min_range_val (float): Minimal value overwriting the colorbar scale
        max_range_val (float): Maximuma value overwriting colorbar scale

    Returns:
        go.Figure: UMAP projection plot
    """
    # print(bool(bsite_restricter),bsite_restricter)
    restrict = False
    discrete_colors_bands = {
        "d-d": "#0207A2",
        "d-f": "#0810FC",
        "d-p": "#5D63FD",
        "d-s": "#B8BBFE",
        "f-d": "#02AB06",
        "f-f": "#55BB45",
        "f-p": "#80CB70",
        "f-s": "#A6DA98",
        "p-d": "#DE8404",
        "p-f": "#E99A42",
        "p-p": "#F1B06D",
        "p-s": "#F8C696",
        "s-d": "#F7398F",
        "s-f": "#FD69A2",
        "s-p": "#FF8EB6",
        "s-s": "#FFB0CA",
        "Vac-d": "#632A02",
        "Vac-f": "#824D2D",
        "Vac-p": "#9F7257",
        "Vac-s": "#BC9984",
    }

    col_to_legend_dict = {
        "bandgap": "Band Gap (eV)",
        "popdiff.total": "Net Spin",
    }

    min_range_val = np.float64(min_range_val)
    max_range_val = np.float64(max_range_val)

    if bsite_restricter:
        bfilt_input_string = str(bsite_restricter).strip().split(",")
        restrict = True
    elif bsite_filter:
        bfilt_input_string = str(bsite_filter).strip().split(",")
    else:
        bfilt_input_string = []

    if len(bfilt_input_string) == 1:
        bfilt_input_string = bfilt_input_string[0].split(" ")

    if (
        len(bfilt_input_string) == 0
        or bfilt_input_string[0] == "None"
        or bfilt_input_string[0] == ""
    ):
        belement_list = set(
            list(dinfo["element.B1"].unique()) + list(dinfo["element.B2"].unique())
        )
    else:
        belement_list = [x.strip().capitalize() for x in bfilt_input_string]
    # print(belement_list)
    # print(restrict)
    if restrict:
        bfiltered_index = list(
            dinfo[
                (dinfo["element.B1"].isin(belement_list))
                & dinfo["element.B2"].isin(belement_list)
            ].index
        )
    else:
        hits1 = list(dinfo[dinfo["element.B1"].isin(belement_list)].index)
        hits2 = list(dinfo[dinfo["element.B2"].isin(belement_list)].index)
        bfiltered_index = list(set(hits1 + hits2))
    bfilt = dinfo.loc[bfiltered_index]
    filtered_index = list(bfilt[bfilt["element.X"].isin(list(xsite_filter))].index)

    dplot["x"] = dumap[selected_data, selected_metric, str(nn_value), "x"]
    dplot["y"] = dumap[selected_data, selected_metric, str(nn_value), "y"]
    if dplot[selected_column].dtype == np.float64:
        colorcolumn = (
            dplot[selected_column]
            if not color_scale == "Log"
            else np.log10(np.abs(dplot[selected_column]))
        )

        if dplot[selected_column].name in col_to_legend_dict.keys():
            colorcolumn.name = col_to_legend_dict[dplot[selected_column].name]
        else:
            colorcolumn.name = dplot[selected_column].name

        if color_scale == "Log":
            # %dplot[f'log(|{selected_column}|)'] = np.log(np.abs(dplot[selected_column]))
            orig_min = (
                np.log10(min_range_val)
                if not np.isnan(min_range_val)
                else colorcolumn[~colorcolumn.isin([np.nan, -np.inf, np.inf])].min()
            )
            orig_max = (
                np.log10(max_range_val)
                if not np.isnan(max_range_val)
                else colorcolumn[~colorcolumn.isin([np.nan, -np.inf, np.inf])].max()
            )

            # orig_min = minrange
            # orig_max = maxrange
            # print(orig_min,orig_max,)
            # tick_values = np.logspace(np.log10(orig_min), np.log10(orig_max), num=10, endpoint=True)
            tick_values1 = list(
                np.linspace(10**orig_min, 10 ** (orig_max / 3), num=4, endpoint=True)
            )
            tick_values2 = list(
                np.linspace(
                    10 ** (orig_max / 3), 10 ** (2 * orig_max / 3), num=3, endpoint=True
                )
            )
            tick_values3 = list(
                np.linspace(
                    10 ** (2 * orig_max / 3), 10**orig_max, num=3, endpoint=True
                )
            )
            tick_values = tick_values1 + tick_values2 + tick_values3
            # tick_values = [round(val, 2) for val in tick_values]
            colorbar = dict(
                tickvals=[np.log10(x) for x in tick_values],
                ticktext=[f"{v:.2g}" for v in tick_values],
            )
            colorrange = [orig_min - 0.1, orig_max + 0.1]
            # colorrange = [orig_min,orig_max]
            # colorcolumn = np.log10(colorcolumn)
            # colorcolumn = np.log10(colorcolumn)
        else:
            orig_min = (
                min_range_val
                if not np.isnan(min_range_val)
                else np.floor(colorcolumn.dropna().min())
            )
            orig_max = (
                max_range_val
                if not np.isnan(max_range_val)
                else np.ceil(colorcolumn.dropna().max())
            )

            tick_values = np.linspace(orig_min, orig_max, num=6, endpoint=True)
            # print(tick_values)
            tick_values = [round(val, 2) for val in tick_values]
            colorbar = dict(
                tickvals=tick_values,
                ticktext=[f"{v:.2f}" for v in tick_values],
            )
            colorrange = [orig_min - 0.1, orig_max + 0.1]

    else:
        colorcolumn = dplot[selected_column]
        colorrange = None
        colorbar = {}
    # print(colorcolumn.name, colorbar['tickvals'],colorbar['ticktext'])
    # if color_scale=='Linear' else dplot[f'log(|{selected_column}|)']

    dfilt = dplot.loc[filtered_index]
    dnegative = dplot.drop(filtered_index)

    if set(colorcolumn.unique().tolist()) <= set(discrete_colors_bands.keys()):
        discretecolormap = discrete_colors_bands
    else:
        discretecolormap = {}

    # print(dfilt)
    # umapfig = px.scatter(dplot, x='x',y='y', opacity=0.3,hover_data=['comp_name_full','element.X','popdiff.total'], hover_name=dplot.index)
    umapfig = px.scatter(
        dfilt,
        x="x",
        y="y",
        color=colorcolumn.loc[filtered_index],
        color_continuous_scale="Plasma",
        color_discrete_map=discretecolormap,
        category_orders={
            "transition_bands": list(discrete_colors_bands.keys()),
            "block_pairing": list(discrete_colors_bands.keys()),
            "block.B1": ["s", "p", "d", "f"],
            "block.B2": ["s", "p", "d", "f", "Vac"],
            "element.X": ["F", "Cl", "Br", "I"],
        },
        hover_data=["comp_name_full", "element.X", "popdiff.total"],
        range_color=colorrange,
        #  colorbar = colorbar,
        hover_name=dplot.loc[filtered_index].index,
    )
    umapfig.add_trace(
        go.Scatter(
            x=dnegative["x"],
            y=dnegative["y"],
            mode="markers",
            marker={"color": "grey", "opacity": 0.4, "size": 3.5},
            showlegend=False,
            hovertext=dnegative.index.to_list(),
        )
    )
    umapfig.update_layout(
        xaxis_title="UMAP 1", xaxis=dict(title=dict(font=dict(size=17)))
    )
    umapfig.update_xaxes(showticklabels=False)
    umapfig.update_layout(
        yaxis_title="UMAP 2", yaxis=dict(title=dict(font=dict(size=17)))
    )
    umapfig.update_yaxes(showticklabels=False)
    # umapfig.update_layout(title=f"UMAP projection of (B1up-B1down) + (B2up - B2down), mindist:{MINDIST}, METRIC:{METRIC}")
    # umapfig.update_layout(activeselection = {'fillcolor':'springgreen'},unselected={"marker": {"opacity": 0.3}},)
    umapfig.update_layout(
        clickmode="event",
        legend=dict(font=dict(size=18), title=dict(font=dict(size=18))),
        coloraxis_colorbar=dict(
            title=colorcolumn.name, tickfont=dict(size=18), **colorbar
        ),
    )
    return umapfig


if __name__ == "__main__":
    app.run(debug=True)
