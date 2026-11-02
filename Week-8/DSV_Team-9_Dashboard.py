import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Spotify Music Intelligence",
    page_icon="🎧",
    layout="wide"
)

FEATURES = [
    "danceability", "energy", "loudness",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]

GREEN = "#1ED760"

st.markdown("""
<style>
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
}
[data-testid="stAppViewContainer"] {
    background: #0b1020;
    color: #f3f6fa;
}
[data-testid="stMetric"] {
    background: #152337;
    border: 1px solid #2a4055;
    border-radius: 14px;
    padding: 18px;
}
[data-testid="stForm"] {
    background: #111c2e;
    border-radius: 16px;
    padding: 22px;
}
.hero {
    background: linear-gradient(110deg, #133e29, #122033);
    border: 1px solid #2d6040;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 25px;
}
.hero h1 {color: white;}
.hero p {color: #c8d7d0;}
</style>
""", unsafe_allow_html=True)

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    GREEN, "#66C7FF", "#FFC857", "#B494FF"
]


def show_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e7edf5"),
        margin=dict(l=20, r=20, t=60, b=30),
        height=420
    )
    st.plotly_chart(fig, width="stretch")


@st.cache_data
def prepare_data(file):
    data = pd.read_csv(file, low_memory=False)
    data.columns = data.columns.str.strip()

    required = ["name", "popularity"] + FEATURES
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    original_rows = len(data)

    for column in ["popularity"] + FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[["popularity"] + FEATURES] = (
        data[["popularity"] + FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    data = data[data["popularity"].between(0, 100)].copy()

    data["name"] = data["name"].fillna("Unknown track").astype(str)

    dates = data.get(
        "album.release_date",
        pd.Series(index=data.index, dtype="object")
    )

    data["release_year"] = pd.to_numeric(
        dates.astype(str).str.extract(r"^(\d{4})", expand=False),
        errors="coerce"
    )

    duration = data.get(
        "duration_ms.x",
        data.get(
            "duration_ms.y",
            pd.Series(np.nan, index=data.index)
        )
    )

    data["duration_minutes"] = (
        pd.to_numeric(duration, errors="coerce") / 60000
    )

    data["album_format"] = data.get(
        "album.album_type",
        pd.Series("Unknown", index=data.index)
    ).fillna("Unknown").astype(str)

    content = data.get(
        "explicit",
        pd.Series("Unknown", index=data.index)
    )

    data["content_label"] = (
        content.astype(str).str.lower()
        .map({"true": "Explicit", "false": "Clean"})
        .fillna("Unknown")
    )

    data = data.reset_index(drop=True)
    data["record_number"] = data.index + 1
    data["track_label"] = (
        data["name"] + " · #" + data["record_number"].astype(str)
    )

    return data, original_rows


st.markdown("""
<div class="hero">
    <p>TEAM 9 · DATA SCIENCE & VISUALIZATION</p>
    <h1>🎧 Spotify Music Intelligence</h1>
    <p>From popularity patterns to musical characteristics:
    explore the story inside our Spotify dataset.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Your dataset")

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "DSV_Team-9_Cleaned_Dataset.csv"
)

try:
    data, original_rows = prepare_data(DATASET_PATH)
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()

if data.empty:
    st.warning("No valid popularity scores between 0 and 100 were found.")
    st.stop()

st.sidebar.success("Your dataset is loaded")
st.sidebar.write(f"Uploaded records: {original_rows:,}")
st.sidebar.write(f"Eligible records: {len(data):,}")

st.header("🔎 Global filters")
st.caption("These filters control every section of the dashboard.")

with st.form("global_filters"):
    known_years = data["release_year"].dropna()
    selected_years = None

    if (
        not known_years.empty
        and known_years.min() < known_years.max()
    ):
        selected_years = st.slider(
            "Release-year range",
            int(known_years.min()),
            int(known_years.max()),
            (
                int(known_years.min()),
                int(known_years.max())
            )
        )

    left, right = st.columns(2)

    with left:
        popularity_range = st.slider(
            "Popularity range", 0, 100, (0, 100)
        )

        album_options = sorted(data["album_format"].unique())

        selected_formats = st.multiselect(
            "Album format",
            album_options,
            default=album_options
        )

        energy_range = st.slider(
            "Energy range", 0.0, 1.0, (0.0, 1.0), step=0.05
        )

    with right:
        content_options = sorted(data["content_label"].unique())

        selected_content = st.multiselect(
            "Content label",
            content_options,
            default=content_options
        )

        danceability_range = st.slider(
            "Danceability range",
            0.0, 1.0, (0.0, 1.0), step=0.05
        )

        search = st.text_input("Search for a song")

    include_unknown_years = st.checkbox(
        "Include unknown release years", value=True
    )

    include_missing_audio = st.checkbox(
        "Include missing energy or danceability", value=True
    )

    st.form_submit_button("Apply filters", type="primary")

mask = (
    data["popularity"].between(*popularity_range)
    & data["album_format"].isin(selected_formats)
    & data["content_label"].isin(selected_content)
)

for column, limits in [
    ("energy", energy_range),
    ("danceability", danceability_range)
]:
    mask &= (
        data[column].between(*limits)
        | (data[column].isna() & include_missing_audio)
    )

if selected_years is not None:
    mask &= (
        data["release_year"].between(*selected_years)
        | (data["release_year"].isna() & include_unknown_years)
    )
elif not include_unknown_years:
    mask &= data["release_year"].notna()

if search:
    mask &= data["name"].str.contains(
        search, case=False, regex=False, na=False
    )

filtered = data[mask].copy()

if filtered.empty:
    st.warning("No tracks match. Broaden your filters and click Apply filters.")
    st.stop()

st.caption(
    f"Showing {len(filtered):,} of {len(data):,} eligible records."
)

st.header("📊 Music collection overview")

a, b, c, d = st.columns(4)

a.metric("Track records", f"{len(filtered):,}")
b.metric("Average popularity", f"{filtered['popularity'].mean():.1f}")

duration_median = filtered["duration_minutes"].median()

c.metric(
    "Median duration",
    f"{duration_median:.2f} min"
    if pd.notna(duration_median) else "Unavailable"
)

d.metric(
    "Popularity ≥ 80",
    f"{filtered['popularity'].ge(80).mean():.1%}"
)

a, b, c = st.columns(3)

a.metric("Average energy", f"{filtered['energy'].mean():.2f}")
b.metric(
    "Average danceability",
    f"{filtered['danceability'].mean():.2f}"
)
c.metric(
    "Explicit records",
    f"{filtered['content_label'].eq('Explicit').sum():,}"
)

st.caption(
    "Popularity comes from your dataset, not live Spotify data. "
    "The ≥80 threshold is defined for this dashboard."
)

tabs = st.tabs([
    "1 · Popularity story",
    "2 · Release & format",
    "3 · Audio analysis",
    "4 · PCA",
    "5 · Compare tracks",
    "6 · Findings & data"
])

with tabs[0]:
    st.header("Which tracks attract higher popularity scores?")
    st.write(
        "Begin with the overall distribution, then examine "
        "the highest-ranked track records."
    )

    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            filtered,
            x="popularity",
            nbins=25,
            title="How is popularity distributed?",
            labels={"popularity": "Popularity score"}
        )
        fig.update_layout(yaxis_title="Track records")
        show_chart(fig)

        st.caption(
            f"Median popularity: "
            f"{filtered['popularity'].median():.1f}. "
            f"The middle 50% lies between "
            f"{filtered['popularity'].quantile(.25):.1f} and "
            f"{filtered['popularity'].quantile(.75):.1f}."
        )

    with right:
        top_n = st.slider("Number of leading tracks", 5, 20, 10)

        top = (
            filtered.nlargest(top_n, "popularity")
            .sort_values("popularity")
        )

        fig = px.bar(
            top,
            x="popularity",
            y="track_label",
            orientation="h",
            title=f"Top {top_n} track records",
            labels={
                "track_label": "Track",
                "popularity": "Popularity"
            }
        )
        show_chart(fig)

    bands = pd.cut(
        filtered["popularity"],
        bins=[-0.01, 39, 59, 79, 100],
        labels=["0–39", "40–59", "60–79", "80–100"]
    )

    band_counts = (
        bands.value_counts(sort=False)
        .rename_axis("Popularity band")
        .reset_index(name="Records")
    )

    left, right = st.columns(2)

    with left:
        fig = px.pie(
            band_counts,
            names="Popularity band",
            values="Records",
            hole=0.6,
            title="Popularity-band composition"
        )
        show_chart(fig)

    with right:
        fig = px.histogram(
            filtered,
            x="duration_minutes",
            nbins=35,
            title="How long are the tracks?",
            labels={"duration_minutes": "Duration (minutes)"}
        )
        show_chart(fig)

    st.caption(
        "Popularity bands are descriptive groups, not official "
        "Spotify classifications."
    )

with tabs[1]:
    st.header("How is the collection represented across releases?")

    annual = (
        filtered.dropna(subset=["release_year"])
        .groupby("release_year")
        .agg(
            Records=("popularity", "size"),
            Average_popularity=("popularity", "mean")
        )
        .reset_index()
    )

    left, right = st.columns(2)

    with left:
        if not annual.empty:
            fig = px.line(
                annual,
                x="release_year",
                y="Records",
                markers=True,
                title="Track records by release year",
                labels={"release_year": "Release year"}
            )
            show_chart(fig)
        else:
            st.info("No release years are available.")

    with right:
        composition = (
            filtered.groupby(["album_format", "content_label"])
            .size()
            .reset_index(name="Records")
        )

        fig = px.bar(
            composition,
            x="album_format",
            y="Records",
            color="content_label",
            barmode="group",
            title="Album format and content labels",
            labels={
                "album_format": "Album format",
                "content_label": "Content"
            }
        )
        show_chart(fig)

    st.caption(
        "Release-year counts describe dataset coverage, not all "
        "Spotify releases. Popularity is not measured at release."
    )

    fig = px.box(
        filtered,
        x="album_format",
        y="popularity",
        color="album_format",
        points=False,
        title="Popularity spread across album formats",
        labels={"album_format": "Album format"}
    )
    show_chart(fig)

    st.subheader("Compare two release years")

    available_years = sorted(
        filtered["release_year"].dropna().astype(int).unique()
    )

    if len(available_years) >= 2:
        left, right = st.columns(2)

        year_a = left.selectbox(
            "First year", available_years,
            index=len(available_years) - 2
        )
        year_b = right.selectbox(
            "Second year", available_years,
            index=len(available_years) - 1
        )

        rows = []

        for year in [year_a, year_b]:
            group = filtered[filtered["release_year"].eq(year)]

            rows.append({
                "Release year": int(year),
                "Track records": len(group),
                "Average popularity": group["popularity"].mean(),
                "Average energy": group["energy"].mean(),
                "Average danceability": group["danceability"].mean(),
                "Median duration (min)": group["duration_minutes"].median()
            })

        st.dataframe(
            pd.DataFrame(rows).round(2),
            hide_index=True,
            width="stretch"
        )
    else:
        st.info("Include at least two release years to compare them.")

with tabs[2]:
    st.header("Which audio features relate to popularity?")

    feature = st.selectbox("Choose an audio feature", FEATURES)

    valid = filtered.dropna(subset=[feature])

    sample = valid.sample(
        n=min(3500, len(valid)),
        random_state=42
    )

    fig = px.scatter(
        sample,
        x=feature,
        y="popularity",
        color="popularity",
        hover_name="name",
        opacity=0.5,
        color_continuous_scale="Greens",
        title=f"{feature.title()} versus popularity"
    )
    show_chart(fig)

    st.caption(
        f"Scatter plot displays {len(sample):,} sampled records. "
        "The correlation calculations use all available pairs."
    )

    correlations = (
        filtered[FEATURES]
        .corrwith(filtered["popularity"])
        .dropna()
        .sort_values()
    )

    left, right = st.columns(2)

    with left:
        if not correlations.empty:
            correlation_table = (
                correlations.rename_axis("Feature")
                .reset_index(name="Pearson correlation")
            )

            fig = px.bar(
                correlation_table,
                x="Pearson correlation",
                y="Feature",
                orientation="h",
                range_x=[-1, 1],
                title="Linear associations with popularity"
            )
            show_chart(fig)
        else:
            st.info("More variation is needed for correlations.")

    with right:
        fig = px.imshow(
            filtered[FEATURES].corr(),
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Relationships among audio features"
        )
        show_chart(fig)

    st.caption(
        "Correlation describes association, not causation. "
        "Missing values are handled using available pairs."
    )

    st.subheader("Do audio profiles differ by popularity group?")

    comparison_rows = []
    profile_features = [
        "danceability", "energy", "acousticness",
        "instrumentalness", "valence"
    ]

    for label, group in [
        ("Below 50", filtered[filtered["popularity"] < 50]),
        ("At least 80", filtered[filtered["popularity"] >= 80])
    ]:
        if not group.empty:
            for column in profile_features:
                comparison_rows.append({
                    "Group": f"{label} (n={len(group):,})",
                    "Feature": column,
                    "Mean": group[column].mean()
                })

    if comparison_rows:
        fig = px.bar(
            pd.DataFrame(comparison_rows),
            x="Feature",
            y="Mean",
            color="Group",
            barmode="group",
            range_y=[0, 1],
            title="Average musical profiles"
        )
        show_chart(fig)

    st.caption(
        "Scores from 50 to below 80 are excluded from this comparison. "
        "Differences are descriptive; no significance test is implied."
    )

with tabs[3]:
    st.header("Can fewer dimensions summarise the music?")
    st.write(
        "PCA combines the audio features into principal components. "
        "Popularity is used only to colour the resulting map."
    )

    pca_data = filtered[FEATURES].dropna(axis=1, how="all")
    missing_cells = int(pca_data.isna().sum().sum())

    pca_data = pca_data.fillna(pca_data.median())
    pca_data = pca_data.loc[:, pca_data.nunique() > 1]

    if len(pca_data) < 3 or pca_data.shape[1] < 2:
        st.info(
            "PCA needs at least three records and two varying features. "
            "Broaden your filters."
        )
    else:
        scaled = StandardScaler().fit_transform(pca_data)

        model = PCA()
        scores = model.fit_transform(scaled)

        variance = model.explained_variance_ratio_ * 100
        cumulative = np.cumsum(variance)

        components_needed = min(
            int(np.searchsorted(cumulative, 90) + 1),
            len(variance)
        )

        a, b, c = st.columns(3)

        a.metric("Usable features", pca_data.shape[1])
        b.metric("Components for 90%", components_needed)
        c.metric(
            "Variance in PC1 + PC2",
            f"{variance[:2].sum():.1f}%"
        )

        fig = go.Figure()

        components = list(range(1, len(variance) + 1))

        fig.add_bar(
            x=components,
            y=variance,
            name="Individual variance",
            marker_color=GREEN
        )

        fig.add_scatter(
            x=components,
            y=cumulative,
            name="Cumulative variance",
            mode="lines+markers",
            line_color="#66C7FF"
        )

        fig.add_hline(y=90, line_dash="dash")

        fig.update_layout(
            title="Explained variance by component",
            xaxis_title="Principal component",
            yaxis_title="Explained variance (%)"
        )
        show_chart(fig)

        component_names = [
            f"PC{i + 1}" for i in range(scores.shape[1])
        ]

        left, right = st.columns(2)

        pc_x = left.selectbox(
            "Horizontal component", component_names
        )

        pc_y = right.selectbox(
            "Vertical component",
            [name for name in component_names if name != pc_x]
        )

        mapped = pd.DataFrame(scores, columns=component_names)
        mapped["name"] = filtered["name"].to_numpy()
        mapped["popularity"] = filtered["popularity"].to_numpy()

        mapped = mapped.sample(
            n=min(3500, len(mapped)),
            random_state=42
        )

        fig = px.scatter(
            mapped,
            x=pc_x,
            y=pc_y,
            color="popularity",
            hover_name="name",
            opacity=0.55,
            color_continuous_scale="Greens",
            title="Tracks in principal-component space",
            labels={
                pc_x: (
                    f"{pc_x} "
                    f"({variance[component_names.index(pc_x)]:.1f}%)"
                ),
                pc_y: (
                    f"{pc_y} "
                    f"({variance[component_names.index(pc_y)]:.1f}%)"
                )
            }
        )
        show_chart(fig)

        coefficients = pd.DataFrame(
            model.components_.T,
            index=pca_data.columns,
            columns=component_names
        )

        fig = px.imshow(
            coefficients,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Feature contributions: component coefficients"
        )
        show_chart(fig)

        st.caption(
            f"PCA fills {missing_cells:,} missing audio cells with "
            "selection medians and standardises features before fitting. "
            "It is recalculated for the active selection, so component "
            "directions and results can differ from your saved Week 7 PCA. "
            "Nearby points are similar in the displayed dimensions only."
        )

        st.download_button(
            "Download PCA coefficients",
            coefficients.to_csv().encode("utf-8"),
            file_name="spotify_pca_coefficients.csv"
        )

with tabs[4]:
    st.header("Compare two track records")

    options = filtered["record_number"].tolist()

    labels = dict(zip(
        filtered["record_number"],
        filtered["track_label"]
    ))

    left, right = st.columns(2)

    track_a = left.selectbox(
        "First track",
        options,
        format_func=lambda value: labels[value]
    )

    track_b = right.selectbox(
        "Second track",
        options,
        index=min(1, len(options) - 1),
        format_func=lambda value: labels[value]
    )

    selected = (
        filtered.set_index("record_number")
        .loc[[track_a, track_b]]
        .copy()
    )

    selected["Selection"] = ["Track A", "Track B"]

    radar_features = [
        "danceability", "energy", "acousticness",
        "instrumentalness", "liveness", "valence"
    ]

    fig = go.Figure()

    for _, row in selected.iterrows():
        values = [row[column] for column in radar_features]

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=radar_features + [radar_features[0]],
            fill="toself",
            name=row["Selection"]
        ))

    fig.update_layout(
        title="Compare musical profiles",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        )
    )
    show_chart(fig)

    st.dataframe(
        selected[
            ["Selection", "name", "popularity", "duration_minutes"]
            + FEATURES
        ].round(3),
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Radar features share a 0–1 scale. Loudness and tempo use "
        "different units and therefore appear only in the table. "
        "Missing values leave gaps."
    )

with tabs[5]:
    st.header("What does the evidence reveal?")

    st.write(
        f"**Collection:** Your selection contains "
        f"{len(filtered):,} track records, with average popularity "
        f"{filtered['popularity'].mean():.1f} and median popularity "
        f"{filtered['popularity'].median():.1f}."
    )

    if not correlations.empty:
        strongest = correlations.abs().idxmax()

        st.write(
            f"**Audio relationship:** The largest absolute linear "
            f"association with popularity is **{strongest}**, "
            f"with **r = {correlations[strongest]:.3f}**. "
            "The strongest relationship here may still be weak."
        )

    st.subheader("Data-quality checks")

    st.write(
        f"Uploaded records: **{original_rows:,}**. "
        f"Excluded for invalid or out-of-range popularity: "
        f"**{original_rows - len(data):,}**."
    )

    if "id" in data.columns:
        st.write(
            f"Repeated track IDs beyond the first occurrence: "
            f"**{data['id'].duplicated().sum():,}**. "
            "These records are retained."
        )

    quality = pd.DataFrame({
        "Missing values": filtered[FEATURES].isna().sum(),
        "Valid values": filtered[FEATURES].notna().sum(),
        "Distinct values": filtered[FEATURES].nunique()
    })

    st.dataframe(quality, width="stretch")

    st.subheader("Interpret the dashboard correctly")

    st.markdown("""
- Findings describe your uploaded collection, not the whole Spotify catalogue.
- Popularity is a dataset snapshot rather than a live measurement.
- Release-year comparisons do not show historical changes in popularity.
- Correlation does not establish causation.
- PCA summarises audio variation; it does not predict popularity.
- No genre or stream-count analysis is included because those fields are absent.
- Repeated records can influence summaries.
""")

    st.subheader("Presentation narrative")

    st.write(
        "First, introduce the music collection and popularity landscape. "
        "Next, explore release coverage and audio-feature relationships. "
        "Then explain how PCA summarises the musical features. "
        "Finally, compare two tracks and discuss findings and limitations."
    )

    with st.expander("View filtered data"):
        st.dataframe(
            filtered,
            hide_index=True,
            width="stretch"
        )

    st.download_button(
        "Download filtered dataset",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="spotify_filtered.csv",
        mime="text/csv"
    )
