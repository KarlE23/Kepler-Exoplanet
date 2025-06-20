# Using Dash frameworks to make an interactive dashboard. https://dash.plotly.com/ and
# https://www.dash-bootstrap-components.com/
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc


df = pd.read_csv("Kepler_cumulative_cleaned.csv")

# Dropping the error columns and the unnecessary identifiers.
columns_to_drop = [col for col in df.columns if '_err' in col]
columns_to_drop += ['kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_tce_delivname']
df_clean_kepler = df.drop(columns=columns_to_drop, errors="ignore")

# Make them numeric (CONFIRMED=1, FALSE POSITIVE=0), and exclude CANDIDATE.
df_clean_kepler['koi_disposition'] = df_clean_kepler['koi_disposition'].map({'CONFIRMED': 1,'FALSE POSITIVE': 0,
    'CANDIDATE': None
})
binary_disposition_df = df_clean_kepler[df_clean_kepler['koi_disposition'].isin([0, 1])]


# Filter the top habitable exoplanets and selecting some columns that I think is best to describe their values in
# the kepler dataset.
top_habitable_planets = df[
    (df['koi_disposition'] == "CONFIRMED") &
    (df['koi_teq'].between(200, 350)) &
    (df['koi_insol'].between(0.3, 1.5))
][['kepler_name', 'koi_teq', 'koi_insol', 'koi_prad', 'koi_period', 'koi_score']]

top_habitable_planets = top_habitable_planets.sort_values(by='koi_score', ascending=False).head(10)


# Initialize the Dash app and using bootstrap theme for the styling.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# First chart: KOI Score vs Planet Number.
chart_score_vs_planet = px.scatter(
    binary_disposition_df,
    x="koi_score",
    y="koi_tce_plnt_num",
    color="koi_disposition",
    color_continuous_scale=['red', 'green'],
    labels={"koi_disposition": "Disposition (0=FP, 1=Confirmed)"},
    title="KOI Score vs Planet Number"
)

# Second chart: KOI Score vs Signal-to-Noise Ratio.
chart_score_vs_snr = px.scatter(
    binary_disposition_df,
    x="koi_score",
    y="koi_model_snr",
    color="koi_disposition",
    color_continuous_scale=['red', 'green'],
    labels={"koi_disposition": "Disposition (0=FP, 1=Confirmed)"},
    title="KOI Score vs Signal-to-Noise Ratio"
)

# Third chart: Insolation vs Temperature for Confirmed Planets and making the habitable zone/band
confirmed_only = df[df["koi_disposition"] == "CONFIRMED"]

chart_insolation_temp = px.scatter(
    confirmed_only,
    x="koi_insol",
    y="koi_teq",
    hover_data=["kepler_name"],
    title="Confirmed Planets: Insolation vs. Temperature"
)
chart_insolation_temp.update_layout(xaxis_type="log")
chart_insolation_temp.add_vrect(x0=0.3, x1=1.5, fillcolor="green", opacity=0.1, line_width=0)
chart_insolation_temp.add_hrect(y0=200, y1=350, fillcolor="purple", opacity=0.1, line_width=0)

# Forth chart: Orbital Period vs Transit Duration.
chart_period_duration = px.scatter(
    confirmed_only,
    x="koi_period",
    y="koi_duration",
    hover_data=["kepler_name"],
    title="Confirmed Planets: Period vs Duration"
)
chart_period_duration.update_layout(xaxis_type="log", yaxis_type="log")
chart_period_duration.add_vrect(x0=50, x1=500, fillcolor="green", opacity=0.1, line_width=0)
chart_period_duration.add_hrect(y0=5, y1=15, fillcolor="purple", opacity=0.1, line_width=0)


# Defying the app.layout structure and content when you open the app.
app.layout = dbc.Container([
    html.H1("Kepler Exoplanet Dashboard", className="text-center"),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=chart_score_vs_planet), md=6),
        dbc.Col(dcc.Graph(figure=chart_score_vs_snr), md=6)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(figure=chart_insolation_temp), md=6),
        dbc.Col(dcc.Graph(figure=chart_period_duration), md=6)
    ]),

    html.H3("Top 10 Confirmed Habitable Exoplanets", className="text-center"),
    dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in top_habitable_planets.columns],
        data=top_habitable_planets.to_dict("records"),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        page_size=10
    )
], fluid=True)

if __name__ == '__main__':
    app.run(debug=True)