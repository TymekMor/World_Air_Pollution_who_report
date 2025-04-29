import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import MarkerCluster
from matplotlib.colors import ListedColormap
import seaborn as sns
def plot_air_quality_map(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str,
    bins=None,
    labels=None,
    cmap_colors=None
):

    if cmap_colors is None:
        cmap_colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#fdae61", "#d73027", "lightgrey"]

    # Grupy po kraju
    grouped = df.groupby(["iso3", "who_country_name"])[value_col].mean().reset_index()
    grouped["category"] = pd.cut(grouped[value_col], bins=bins, labels=labels)

    # Mapa świata
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[world["name"] != "Antarctica"]
    world["geometry"] = world["geometry"].buffer(0)
    world_merged = world.merge(grouped, how="left", left_on="iso_a3", right_on="iso3")

    # Kategorie braków danych
    world_merged["category"] = world_merged["category"].cat.add_categories("Nie występuje w Bazie Danych")
    world_merged["category"] = world_merged["category"].fillna("Nie występuje w Bazie Danych")

    # Kolory i mapa
    cmap = ListedColormap(cmap_colors)
    fig, ax = plt.subplots(figsize=(20, 10))
    world_merged.plot(column="category", ax=ax, cmap=cmap,
                      legend=True,
                      edgecolor='black',     # <--- add edge color
                      linewidth=0.5,
                      legend_kwds={
                          "title": f"Kategorie {value_col.upper()} (µg/m³)",
                          "bbox_to_anchor": (1.02, 1),
                          "loc": "upper left"
                      })
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

    return world_merged

def plot_region_level_map(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str,
    bins=None,
    labels=None,
    cmap_colors=None
):

    if cmap_colors is None:
        cmap_colors = ["#4575b4", "#91bfdb", "#e0f3f8", "#fdae61", "#d73027", "lightgrey"]

    # Mapa świata bez Antarktydy
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    world = world[world["name"] != "Antarctica"]
    world["geometry"] = world["geometry"].buffer(0)

    # Mapowanie: kraj (ISO3) -> region WHO
    country_region = df[["iso3", "who_region"]].drop_duplicates()

    # Średnia wartość zanieczyszczeń dla regionów
    region_values = df.groupby("who_region")[value_col].mean().reset_index()

    # Łączenie kraju z regionem i wartością regionalną
    country_region_values = country_region.merge(region_values, on="who_region", how="left")

    # Przypisanie do mapy
    world = world.merge(country_region_values, how="left", left_on="iso_a3", right_on="iso3")

    # Kategorie
    world["category"] = pd.cut(world[value_col], bins=bins, labels=labels)
    world["category"] = world["category"].cat.add_categories("Nie występuje w Bazie Danych")
    world["category"] = world["category"].fillna("Nie występuje w Bazie Danych")

    # Tworzymy obrysy regionów WHO przez rozpuszczenie granic krajów
    regions_outline = world.dissolve(by="who_region", as_index=False)

    # Rysowanie mapy
    cmap = ListedColormap(cmap_colors)
    fig, ax = plt.subplots(figsize=(20, 10))
    world.plot(
    column="category",
    ax=ax,
    cmap=cmap,        
    legend=True,
    legend_kwds={
        "title": f"Kategorie {value_col.upper()} (µg/m³)",
        "bbox_to_anchor": (1.02, 1),
        "loc": "upper left"
    }
)
    # Rysujemy grube obrysy regionów WHO
    regions_outline.boundary.plot(ax=ax, color="black", linewidth=1)
    
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_region_boxplots(df_long, region):
    """
    Tworzy 1x3 boxploty dla PM2.5, PM10 i NO2 z trendem średnich rocznych.

    Parametry:
    ----------
    df_long : DataFrame w formacie long
    region : str, np. 'European Region'
    """
    pollutants = ['pm2.5_μg_m3', 'pm10_μg_m3', 'no2_μg_m3']
    titles = ['PM2.5', 'PM10', 'NO₂']

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

    for ax, pollutant, title in zip(axes, pollutants, titles):
        data = df_long[(df_long["who_region"] == region) & (df_long["pollutant"] == pollutant)]

        if data.empty:
            ax.set_title(f"No data for {title}")
            ax.axis("off")
            continue

        # Rysuj boxplot
        sns.boxplot(data=data, x="measurement_year", y="value", ax=ax, color="lightblue")

        # Oblicz średnie roczne i narysuj trend

        ax.set_title(f"{title} in {region}")
        ax.set_xlabel("Year")
        ax.set_ylabel("μg/m³")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()