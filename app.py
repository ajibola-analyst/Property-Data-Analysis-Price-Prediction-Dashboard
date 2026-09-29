import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from src.data_processing import load_and_merge_data, clean_and_transform_data, train_price_model


st.set_page_config(
    page_title="Buenos Aires, Mapped in Real Estate",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


PLASTER = "#F5F0E6"
CARD = "#FFFFFF"
INK = "#2B2621"
MUTED = "#8A7F6E"
JACARANDA = "#6B5B95"
JACARANDA_DARK = "#453A63"
TEAL = "#2C4A52"
BRICK = "#B5563C"
OCHRE = "#C79A46"
BORDER = "#E4DAC7"

TYPE_COLORS = {
    "apartment": JACARANDA,
    "house": TEAL,
    "PH": BRICK,
    "store": OCHRE,
}
SEQUENTIAL_SCALE = [PLASTER, "#D9CFE8", "#B3A2CC", "#8B76AC", JACARANDA_DARK]

REGION_ORDER = ["Capital Federal", "Greater BA — North", "Greater BA — West", "Greater BA — South"]

FONT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

h1, h2, h3, .hero-title { font-family: 'Fraunces', serif; }

.stApp { background-color: %(plaster)s; }

.hero-title {
    font-size: 2.6rem;
    font-weight: 600;
    color: %(ink)s;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.02rem;
    color: %(muted)s;
    max-width: 640px;
}
.kpi-card {
    background-color: %(card)s;
    border: 1px solid %(border)s;
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%%;
}
.kpi-label {
    font-size: 0.82rem;
    color: %(muted)s;
    margin-bottom: 4px;
}
.kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 1.6rem;
    color: %(ink)s;
    font-weight: 600;
}
section[data-testid="stSidebar"] {
    background-color: %(card)s;
    border-right: 1px solid %(border)s;
}
</style>
""" % {"plaster": PLASTER, "ink": INK, "muted": MUTED, "card": CARD, "border": BORDER}

st.markdown(FONT_CSS, unsafe_allow_html=True)


def base_layout(fig, height=420):
    """Shared, quiet chart styling so every chart in the app matches."""
    fig.update_layout(
        height=height,
        plot_bgcolor=CARD,
        paper_bgcolor=CARD,
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        title_font=dict(family="Fraunces, serif", size=17, color=INK),
    )
    fig.update_xaxes(gridcolor=BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=BORDER, zeroline=False)
    return fig


def kpi_card(label, value):
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div></div>""",
        unsafe_allow_html=True,
    )



@st.cache_data
def get_data():
    raw = load_and_merge_data("data/buenos-aires-real-estate-1.csv", "data/buenos-aires-real-estate-2.csv")
    return clean_and_transform_data(raw)


@st.cache_resource
def get_model(df):
    return train_price_model(df)


df = get_data()
model, feature_names, model_metrics = get_model(df)


st.sidebar.markdown("### Filter the listings")

regions_present = [r for r in REGION_ORDER if r in df["region_label"].unique()]
selected_region = st.sidebar.selectbox("Region", ["All regions"] + regions_present)

region_scope = df if selected_region == "All regions" else df[df["region_label"] == selected_region]
neighborhoods = ["All neighborhoods"] + sorted(
    n for n in region_scope["neighborhood"].unique() if n and n != "Unknown"
)
selected_neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods)

prop_types = ["All property types"] + sorted(df["property_type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Property type", prop_types)

min_price, max_price = int(df["price_aprox_usd"].min()), int(df["price_aprox_usd"].max())
price_range = st.sidebar.slider(
    "Price range (USD)", min_price, max_price, (min_price, min(max_price, 500_000)), step=5_000
)

filtered = df.copy()
if selected_region != "All regions":
    filtered = filtered[filtered["region_label"] == selected_region]
if selected_neighborhood != "All neighborhoods":
    filtered = filtered[filtered["neighborhood"] == selected_neighborhood]
if selected_type != "All property types":
    filtered = filtered[filtered["property_type"] == selected_type]
filtered = filtered[filtered["price_aprox_usd"].between(*price_range)]

st.sidebar.markdown(f"**{len(filtered):,}** listings match your filters")


st.markdown('<div class="hero-title">Buenos Aires, mapped in real estate</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">A tour of {len(df):,} property listings across Capital Federal and Greater '
    "Buenos Aires — where things cost more, what a typical listing looks like, and a model that "
    "estimates a fair price from a few basic details.</div>",
    unsafe_allow_html=True,
)
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Map", "Price patterns", "Price estimator", "Explore the data"]
)


# TAB 1 — OVERVIEW

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Listings shown", f"{len(filtered):,}")
    with c2:
        kpi_card("Median price", f"${filtered['price_aprox_usd'].median():,.0f}" if len(filtered) else "—")
    with c3:
        kpi_card(
            "Median price / m²",
            f"${filtered['price_usd_per_m2'].median():,.0f}" if len(filtered) else "—",
        )
    with c4:
        kpi_card(
            "Typical size",
            f"{filtered['surface_covered_in_m2'].median():,.0f} m²" if len(filtered) else "—",
        )

    st.write("")
    left, right = st.columns([1.1, 1])

    with left:
        region_summary = (
            df.groupby("region_label")["price_usd_per_m2"]
            .median()
            .reindex(REGION_ORDER)
            .dropna()
            .sort_values()
        )
        fig = go.Figure(
            go.Bar(
                x=region_summary.values,
                y=region_summary.index,
                orientation="h",
                marker_color=JACARANDA,
                text=[f"${v:,.0f}" for v in region_summary.values],
                textposition="outside",
            )
        )
        fig.update_layout(title="Median price per m² by region")
        fig.update_xaxes(title="USD / m²")
        st.plotly_chart(base_layout(fig, height=320), use_container_width=True)

    with right:
        top_neigh = (
            df[df["neighborhood"] != "Unknown"]
            .groupby("neighborhood")
            .filter(lambda g: len(g) >= 30)
            .groupby("neighborhood")["price_usd_per_m2"]
            .median()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )
        fig2 = go.Figure(
            go.Bar(
                x=top_neigh.values,
                y=top_neigh.index,
                orientation="h",
                marker_color=BRICK,
                text=[f"${v:,.0f}" for v in top_neigh.values],
                textposition="outside",
            )
        )
        fig2.update_layout(title="10 priciest neighborhoods (min. 30 listings)")
        fig2.update_xaxes(title="USD / m²")
        st.plotly_chart(base_layout(fig2, height=320), use_container_width=True)

    st.caption(
        "Prices are asking prices in USD as listed on Properati, not confirmed sale prices."
    )


# TAB 2 — MAP

with tab2:
    st.markdown("#### Where the filtered listings sit on the map")
    map_df = filtered.dropna(subset=["lat", "lon"]).copy()

    if len(map_df):
        view_state = pdk.ViewState(
            latitude=map_df["lat"].median(),
            longitude=map_df["lon"].median(),
            zoom=10.5,
            pitch=0,
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            map_df,
            get_position=["lon", "lat"],
            get_fill_color=[107, 91, 149, 140],
            get_radius=55,
            radius_min_pixels=3,
            radius_max_pixels=9,
            pickable=True,
            stroked=False,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="light",
            tooltip={
                "html": "<b>{property_type}</b> in {neighborhood}<br/>"
                "${price_aprox_usd} · {surface_covered_in_m2} m²",
                "style": {"backgroundColor": TEAL, "color": "white"},
            },
        )
        st.pydeck_chart(deck)
        st.caption(f"Showing {len(map_df):,} of {len(filtered):,} filtered listings that have coordinates.")
    else:
        st.info("No listings with map coordinates match the current filters — try widening them.")


# TAB 3 — PRICE PATTERNS

with tab3:
    left, right = st.columns(2)

    with left:
        st.markdown("#### What a listing typically costs, by property type")
        box_df = filtered[filtered["price_aprox_usd"] < filtered["price_aprox_usd"].quantile(0.97)]
        fig3 = go.Figure()
        for ptype, color in TYPE_COLORS.items():
            vals = box_df.loc[box_df["property_type"] == ptype, "price_aprox_usd"]
            if len(vals):
                fig3.add_trace(
                    go.Box(y=vals, name=ptype, marker_color=color, boxpoints=False)
                )
        fig3.update_layout(showlegend=False, yaxis_title="Price (USD)")
        st.plotly_chart(base_layout(fig3), use_container_width=True)
        st.caption("Top 3% of prices excluded from this view so the boxes stay readable.")

    with right:
        st.markdown("#### Where price and size land together")
        density_df = filtered.dropna(subset=["surface_covered_in_m2", "price_aprox_usd"])
        density_df = density_df[
            (density_df["surface_covered_in_m2"] < 400) & (density_df["price_aprox_usd"] < 800_000)
        ]
        if len(density_df) > 20:
            fig4 = go.Figure(
                go.Histogram2d(
                    x=density_df["surface_covered_in_m2"],
                    y=density_df["price_aprox_usd"],
                    colorscale=SEQUENTIAL_SCALE,
                    nbinsx=30,
                    nbinsy=30,
                    colorbar=dict(title="listings"),
                )
            )
            fig4.update_layout(xaxis_title="Covered area (m²)", yaxis_title="Price (USD)")
            st.plotly_chart(base_layout(fig4), use_container_width=True)
            st.caption("Darker cells = more listings at that size and price — a clearer read than a raw scatter plot.")
        else:
            st.info("Not enough listings in this filter to show the density chart.")

    with st.expander("See how all the numeric fields relate to each other"):
        num_cols = ["price_aprox_usd", "surface_covered_in_m2", "surface_total_in_m2", "price_usd_per_m2", "rooms"]
        corr = filtered[num_cols].corr()
        fig5 = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale=SEQUENTIAL_SCALE,
            aspect="auto",
        )
        st.plotly_chart(base_layout(fig5, height=380), use_container_width=True)


# TAB 4 — PRICE ESTIMATOR

with tab4:
    st.markdown("#### Get a fair-price estimate")
    st.caption(
        f"Trained on {len(df):,} listings · explains about "
        f"{model_metrics['r2']*100:.0f}% of price variation · typically within "
        f"${model_metrics['mae']:,.0f} of the actual price."
    )

    c1, c2, c3 = st.columns(3)
    in_type = c1.selectbox("Property type", sorted(df["property_type"].unique()))
    in_region = c2.selectbox("Region", regions_present)
    in_surface = c3.number_input("Covered area (m²)", min_value=15, max_value=800, value=70, step=5)
    in_rooms = st.slider("Rooms", 1, 8, 3)

    if st.button("Estimate price", type="primary"):
        input_row = pd.DataFrame(0, index=[0], columns=feature_names)
        if "surface_covered_in_m2" in input_row.columns:
            input_row["surface_covered_in_m2"] = in_surface
        if "rooms" in input_row.columns:
            input_row["rooms"] = in_rooms
        for col in (f"property_type_{in_type}", f"region_label_{in_region}"):
            if col in input_row.columns:
                input_row[col] = 1

        prediction = model.predict(input_row)[0]
        lo, hi = max(0, prediction - model_metrics["mae"]), prediction + model_metrics["mae"]

        st.markdown(
            f"""<div class="kpi-card" style="border-left:4px solid {JACARANDA}; max-width:420px;">
            <div class="kpi-label">Estimated fair price</div>
            <div class="kpi-value">${prediction:,.0f}</div>
            <div class="kpi-label" style="margin-top:6px;">Likely range: ${lo:,.0f} – ${hi:,.0f}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# TAB 5 — EXPLORE THE DATA

with tab5:
    st.markdown(f"#### {len(filtered):,} filtered listings")
    display_cols = [
        "operation", "property_type", "region_label", "neighborhood",
        "price_aprox_usd", "surface_covered_in_m2", "price_usd_per_m2",
        "rooms", "properati_url",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download this view as CSV",
        data=csv_data,
        file_name="buenos_aires_real_estate_filtered.csv",
        mime="text/csv",
    )
