from __future__ import annotations

import folium
from folium.plugins import HeatMap


def build_prediction_map_html(prediction_df):
    if prediction_df.empty:
        return "<div>No predictions to plot.</div>"

    points = prediction_df.dropna(subset=["latitude", "longitude"]).copy()
    if points.empty:
        return "<div>No latitude/longitude available for plotting.</div>"

    center_lat = points["latitude"].mean()
    center_lon = points["longitude"].mean()

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=3,
        control_scale=True,
        tiles="OpenStreetMap",
    )

    heat_data = []

    for _, row in points.iterrows():
        popup_html = f"""
        <b>Event ID:</b> {row['trigger_event_id']}<br>
        <b>Prob 24h:</b> {row['prob_24h']:.3f}<br>
        <b>Prob 72h:</b> {row['prob_72h']:.3f}<br>
        <b>Pred Count 24h:</b> {row['count_24h']:.2f}<br>
        <b>Pred Count 72h:</b> {row['count_72h']:.2f}<br>
        <b>Pred Mag 24h:</b> {row['mag_24h']:.2f}<br>
        <b>Pred Mag 72h:</b> {row['mag_72h']:.2f}
        """

        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=popup_html,
            tooltip=row["trigger_event_id"],
        ).add_to(fmap)

        # Weighted by predicted 72h magnitude; with only 1-3 points, this acts like a glow.
        heat_data.append([row["latitude"], row["longitude"], max(row["mag_72h"], 0.0)])

    if heat_data:
        HeatMap(
            heat_data,
            name="Predicted magnitude heatmap",
            radius=35,
            blur=25,
            min_opacity=0.35,
        ).add_to(fmap)

    folium.LayerControl().add_to(fmap)
    return fmap._repr_html_()