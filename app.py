"""
Zonas Gastronómicas de La Paz — App Streamlit
Aprendizaje No Supervisado · HDBSCAN sin one-hot de tipo

Archivo esperado en la misma carpeta:
    dataset_lapaz_clustered.csv

Columnas esperadas principales:
    nombre, tipo, tipo_cat, lat, lon, rating, n_resenas, precio,
    direccion, densidad_500m, cluster_hdbscan, prob_hdbscan,
    umap_1, umap_2
"""

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import st_folium

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Zonas Gastronómicas · La Paz",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("dataset_lapaz_clustered.csv")

TIPO_EMOJI = {
    "restaurante": "🍲",
    "cafe": "☕",
    "bar": "🍺",
    "mercado": "🛒",
    "heladeria": "🍦",
    "comida_rapida": "🌮",
    "otro": "🍽️",
}

TIPO_COLOR = {
    "restaurante": "#E63946",
    "cafe": "#2A9D8F",
    "bar": "#F4A261",
    "mercado": "#457B9D",
    "heladeria": "#A8DADC",
    "comida_rapida": "#6A0572",
    "otro": "#94A3B8",
}

DEFAULT_COLORS = [
    "#2A9D8F", "#E2A830", "#457B9D", "#E63946", "#6A0572",
    "#F4A261", "#94A3B8", "#3D405B", "#A8DADC", "#8ECAE6",
    "#FFB703", "#219EBC", "#FB8500", "#8338EC", "#06D6A0",
]

# Nombres interpretativos sugeridos para una corrida de 9 clusters.
# Si tus IDs cambian o aparecen más/menos clusters, la app igual funciona.
CLUSTER_LABELS_BASE = {
    0:  {"nombre": "Zona Gastronómica General",      "emoji": "🍽️"},
    1:  {"nombre": "Alta Densidad · Precio Alto",    "emoji": "💎"},
    2:  {"nombre": "Mercados Populares",             "emoji": "🛒"},
    3:  {"nombre": "Alta Densidad · Popularidad",    "emoji": "🔥"},
    4:  {"nombre": "Alto Rating · Poco Conocidos",   "emoji": "⭐"},
    5:  {"nombre": "Los Más Populares",              "emoji": "🏆"},
    6:  {"nombre": "Zona Dispersa",                  "emoji": "📍"},
    7:  {"nombre": "Alta Densidad · Precio Bajo",    "emoji": "🏙️"},
    8:  {"nombre": "Baratos y Masivos",              "emoji": "👥"},
    -1: {"nombre": "Outliers",                       "emoji": "·"},
}

# ============================================================
# ESTILOS
# ============================================================
st.markdown(
    """
<style>
  .main-title {
    font-size:2.2rem; font-weight:800; margin-bottom:0;
    background:linear-gradient(135deg,#0D9488,#E2A830);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  }
  .subtitle { color:#64748B; font-size:.92rem; margin-top:2px; }
  .kpi { background:#1E293B; border-radius:10px; padding:.85rem 1rem; border-left:4px solid #0D9488; }
  .kpi-val { font-size:1.72rem; font-weight:700; color:#5EEAD4; }
  .kpi-lbl { font-size:.72rem; color:#94A3B8; text-transform:uppercase; letter-spacing:.05em; }
  .zone-card { padding:8px 10px; border-radius:8px; margin-bottom:7px; border-left:4px solid; background:#F8FAFC; }
  .section-hdr { font-size:1.05rem; font-weight:700; color:#1E293B;
    border-bottom:2px solid #0D9488; padding-bottom:4px; margin-bottom:.9rem; }
  [data-testid="stSidebar"] { background:#0F172A; }
  [data-testid="stSidebar"] * { color:#E2E8F0 !important; }
</style>
""",
    unsafe_allow_html=True,
)
import re

def extraer_precio_promedio(texto):
    if pd.isna(texto):
        return np.nan

    texto = str(texto).replace("\xa0", " ").strip()

    if texto in ["", "nan", "N/D"]:
        return np.nan

    if texto == "$":
        return 10
    if texto == "$$":
        return 50
    if texto == "$$$":
        return 140
    if texto == "$$$$":
        return 250

    nums = re.findall(r"\d+", texto)

    if "Más de" in texto and len(nums) >= 1:
        minimo = int(nums[0])
        return minimo + 50

    if len(nums) >= 2:
        minimo = int(nums[0])
        maximo = int(nums[1])
        return (minimo + maximo) / 2

    if len(nums) == 1:
        return float(nums[0])

    return np.nan
# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(
            f"No se encontró el archivo `{path}`. Coloca `dataset_lapaz_clustered.csv` "
            "en la misma carpeta que `app.py`."
        )
        st.stop()

    df = pd.read_csv(path)

    required = ["nombre", "lat", "lon", "rating", "n_resenas", "cluster_hdbscan"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas obligatorias en el CSV: {missing}")
        st.stop()

    # Columnas opcionales con valores seguros
    if "tipo" not in df.columns:
        df["tipo"] = "Sin dato"
    if "tipo_cat" not in df.columns:
        df["tipo_cat"] = "otro"
    if "precio" not in df.columns:
        df["precio"] = "N/D"
    if "direccion" not in df.columns:
        df["direccion"] = ""
    if "densidad_500m" not in df.columns:
        df["densidad_500m"] = 0
    if "precio_num" not in df.columns:
        if "precio_prom" in df.columns:
            df["precio_num"] = df["precio_prom"]
        else:
            df["precio_num"] = df["precio"].apply(extraer_precio_promedio)
    if "prob_hdbscan" not in df.columns:
        df["prob_hdbscan"] = np.nan

    # Si no hay UMAP, se usan lon/lat como respaldo para no romper la app.
    if "umap_1" not in df.columns:
        df["umap_1"] = df["lon"]
    if "umap_2" not in df.columns:
        df["umap_2"] = df["lat"]

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["n_resenas"] = pd.to_numeric(df["n_resenas"], errors="coerce").fillna(0)
    df["densidad_500m"] = pd.to_numeric(df["densidad_500m"], errors="coerce").fillna(0)
    df["precio_num"] = pd.to_numeric(df["precio_num"], errors="coerce")
    mediana_precio = df["precio_num"].median()
    if pd.isna(mediana_precio):
        mediana_precio = 0

    df["precio_num"] = df["precio_num"].fillna(mediana_precio)
    df["cluster_hdbscan"] = pd.to_numeric(df["cluster_hdbscan"], errors="coerce").fillna(-1).astype(int)

    df["tipo_cat"] = df["tipo_cat"].fillna("otro").astype(str)
    df["precio_show"] = df["precio"].fillna("N/D").astype(str)
    df["tipo_emoji"] = df["tipo_cat"].map(lambda t: TIPO_EMOJI.get(t, "🍽️"))

    return df


def build_cluster_metadata(df: pd.DataFrame) -> dict:
    cluster_ids = sorted(df["cluster_hdbscan"].unique())
    clusters = {}
    non_noise = [c for c in cluster_ids if c != -1]

    for i, cl in enumerate(non_noise):
        sub = df[df["cluster_hdbscan"] == cl]
        base = CLUSTER_LABELS_BASE.get(cl, {"nombre": f"Zona {cl}", "emoji": "📍"})
        color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

        rating = sub["rating"].mean()
        densidad = sub["densidad_500m"].mean()
        precio = sub["precio_num"].mean()
        resenas = sub["n_resenas"].mean()
        n = len(sub)

        desc = (
            f"{n} lugares · rating {rating:.2f} · densidad {densidad:.1f}/500m · "
            f"precio Bs {precio:.1f} · {resenas:.0f} reseñas prom."
        )

        clusters[cl] = {
            "nombre": base["nombre"],
            "emoji": base["emoji"],
            "color": color,
            "desc": desc,
        }

    if -1 in cluster_ids:
        n_out = int((df["cluster_hdbscan"] == -1).sum())
        pct_out = n_out / len(df) * 100 if len(df) else 0
        clusters[-1] = {
            "nombre": "Outliers",
            "emoji": "·",
            "color": "#E2E8F0",
            "desc": f"{n_out} lugares sin zona definida ({pct_out:.1f}%)",
        }

    return clusters


def add_cluster_columns(df: pd.DataFrame, clusters: dict) -> pd.DataFrame:
    df = df.copy()
    df["cluster_nombre"] = df["cluster_hdbscan"].map(
        lambda c: clusters.get(c, clusters.get(-1, {"nombre": "Zona"}))["nombre"]
    )
    df["cluster_color"] = df["cluster_hdbscan"].map(
        lambda c: clusters.get(c, clusters.get(-1, {"color": "#999"}))["color"]
    )
    return df


def safe_mean(series: pd.Series, default: str = "0.00") -> str:
    if series.empty:
        return default
    value = series.mean()
    if pd.isna(value):
        return default
    return f"{value:.2f}"


# Cargar datos antes de construir CLUSTERS
df = load_data(DATA_PATH)
CLUSTERS = build_cluster_metadata(df)
df = add_cluster_columns(df, CLUSTERS)

cluster_ids_no_noise = sorted([c for c in df["cluster_hdbscan"].unique() if c != -1])
n_clusters = len(cluster_ids_no_noise)
n_outliers = int((df["cluster_hdbscan"] == -1).sum())
outlier_pct = n_outliers / len(df) * 100 if len(df) else 0

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🗺️ Zonas Gastronómicas")
    st.markdown("**La Paz, Bolivia**")
    st.markdown("---")

    zonas_sel = st.multiselect(
        "ZONAS",
        options=cluster_ids_no_noise,
        default=cluster_ids_no_noise,
        format_func=lambda c: f"{CLUSTERS[c]['emoji']} {CLUSTERS[c]['nombre']}",
    )

    tipos_disp = sorted(df["tipo_cat"].dropna().unique().tolist())
    tipos_sel = st.multiselect(
        "TIPO DE LUGAR",
        options=tipos_disp,
        default=tipos_disp,
        format_func=lambda t: f"{TIPO_EMOJI.get(t, '🍽️')} {t.replace('_', ' ').title()}",
    )

    rmin, rmax = float(df["rating"].min()), float(df["rating"].max())
    rating_rng = st.slider("RATING", rmin, rmax, (rmin, rmax), 0.1)
    show_outliers = st.checkbox("Mostrar outliers", value=False)

    st.markdown("---")
    st.markdown("### Modelo principal")
    st.markdown(
        f"""
        **HDBSCAN**
        - 🗂️ Clusters detectados: **{n_clusters}**
        - 🔇 Outliers: **{outlier_pct:.1f}%**
        - ✅ Sin one-hot de tipo
        - ✅ Tipo usado solo para interpretación posterior
        """
    )

# ============================================================
# FILTRADO
# ============================================================
mask = (
    df["cluster_hdbscan"].isin(zonas_sel)
    & df["tipo_cat"].isin(tipos_sel)
    & df["rating"].between(rating_rng[0], rating_rng[1])
)
if show_outliers:
    mask = mask | (df["cluster_hdbscan"] == -1)

df_f = df[mask].copy()

# ============================================================
# HEADER Y KPIS
# ============================================================
st.markdown('<p class="main-title">Zonas Gastronómicas de La Paz</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Clustering geoespacial no supervisado · HDBSCAN · Sin one-hot de tipo</p>',
    unsafe_allow_html=True,
)

st.success(
    "✅ Versión corregida: el tipo de establecimiento no entra al modelo. "
    "Se usa solo después para interpretar qué descubrieron los clusters."
)

c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    ("Lugares visibles", len(df_f), "#0D9488"),
    ("Zonas visibles", df_f[df_f["cluster_hdbscan"] != -1]["cluster_hdbscan"].nunique(), "#E2A830"),
    ("Rating medio", f"{safe_mean(df_f['rating'])}★", "#E63946"),
    ("Tipos visibles", df_f["tipo_cat"].nunique(), "#6A0572"),
    ("Outliers", f"{outlier_pct:.1f}%", "#94A3B8"),
]

for col, (label, value, color) in zip([c1, c2, c3, c4, c5], kpis):
    with col:
        st.markdown(
            f"""<div class="kpi" style="border-left-color:{color}">
              <div class="kpi-val">{value}</div>
              <div class="kpi-lbl">{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
t_mapa, t_umap, t_analisis, t_modelos, t_datos, t_interp = st.tabs(
    ["🗺️ Mapa", "🔬 UMAP", "📊 Análisis", "⚖️ Modelo", "📋 Datos", "🧠 Interpretación"]
)

# ============================================================
# MAPA
# ============================================================
with t_mapa:
    st.markdown('<div class="section-hdr">Mapa interactivo — Zonas gastronómicas</div>', unsafe_allow_html=True)

    col_map, col_legend = st.columns([3, 1])
    with col_map:
        lat_center = float(df["lat"].mean())
        lon_center = float(df["lon"].mean())
        m = folium.Map(location=[lat_center, lon_center], zoom_start=13, tiles="CartoDB positron")

        ids_to_show = list(zonas_sel) + ([-1] if show_outliers else [])
        for cl in ids_to_show:
            sub = df_f[df_f["cluster_hdbscan"] == cl]
            if sub.empty:
                continue
            info = CLUSTERS.get(cl, CLUSTERS.get(-1))
            fg = folium.FeatureGroup(name=f"{info['emoji']} {info['nombre']}", show=True)

            for _, row in sub.iterrows():
                if pd.isna(row["lat"]) or pd.isna(row["lon"]):
                    continue
                popup = f"""
                <div style="font-family:sans-serif;min-width:190px">
                  <b style="font-size:13px">{row['nombre']}</b><br>
                  <span style="color:#666;font-size:11px">{row['tipo']}</span>
                  <hr style="margin:4px 0">
                  <table style="font-size:11px;width:100%">
                    <tr><td>⭐ Rating</td><td><b>{row['rating']:.1f}</b></td></tr>
                    <tr><td>💬 Reseñas</td><td><b>{int(row['n_resenas'])}</b></td></tr>
                    <tr><td>💰 Precio</td><td><b>{row['precio_show']}</b></td></tr>
                    <tr><td>📍 Zona</td><td><b style="color:{info['color']}">{info['emoji']} {info['nombre']}</b></td></tr>
                  </table>
                  <div style="font-size:10px;color:#999;margin-top:4px">{row.get('direccion', '')}</div>
                </div>
                """
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=6 if cl != -1 else 3,
                    color=info["color"],
                    fill=True,
                    fill_color=info["color"],
                    fill_opacity=0.78 if cl != -1 else 0.3,
                    weight=1.5,
                    popup=folium.Popup(popup, max_width=240),
                    tooltip=f"{info['emoji']} {row['nombre']}",
                ).add_to(fg)
            fg.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        st_folium(m, width=None, height=540, use_container_width=True)

    with col_legend:
        st.markdown("#### Leyenda")
        for cl in cluster_ids_no_noise:
            if cl not in zonas_sel:
                continue
            info = CLUSTERS[cl]
            sub = df[df["cluster_hdbscan"] == cl]
            st.markdown(
                f"""<div class="zone-card" style="border-left-color:{info['color']}">
                  <span style="font-weight:700;font-size:12px">{info['emoji']} {info['nombre']}</span><br>
                  <span style="color:#64748B;font-size:10px">{len(sub)} lugares · {sub['tipo_cat'].nunique()} tipos</span><br>
                  <span style="color:#94A3B8;font-size:9px">{info['desc']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

# ============================================================
# UMAP
# ============================================================
with t_umap:
    st.markdown('<div class="section-hdr">Proyección UMAP</div>', unsafe_allow_html=True)
    col_plot, col_text = st.columns([2, 1])

    with col_plot:
        color_by = st.radio("Colorear por", ["Zona HDBSCAN", "Tipo de lugar", "Rating", "Precio"], horizontal=True)
        fig_u = go.Figure()

        if color_by == "Zona HDBSCAN":
            for cl in sorted(df_f["cluster_hdbscan"].unique()):
                sub = df_f[df_f["cluster_hdbscan"] == cl]
                info = CLUSTERS.get(cl, CLUSTERS.get(-1))
                fig_u.add_trace(
                    go.Scatter(
                        x=sub["umap_1"],
                        y=sub["umap_2"],
                        mode="markers",
                        name=f"{info['emoji']} {info['nombre']}",
                        marker=dict(color=info["color"], size=7, opacity=0.75, line=dict(color="white", width=0.5)),
                        text=sub["nombre"],
                        customdata=np.stack([sub["tipo_cat"], sub["rating"].round(1)], axis=-1),
                        hovertemplate="<b>%{text}</b><br>Tipo: %{customdata[0]}<br>Rating: %{customdata[1]}★<extra></extra>",
                    )
                )

        elif color_by == "Tipo de lugar":
            for tipo in sorted(df_f["tipo_cat"].dropna().unique()):
                sub = df_f[df_f["tipo_cat"] == tipo]
                fig_u.add_trace(
                    go.Scatter(
                        x=sub["umap_1"],
                        y=sub["umap_2"],
                        mode="markers",
                        name=f"{TIPO_EMOJI.get(tipo, '🍽️')} {tipo}",
                        marker=dict(color=TIPO_COLOR.get(tipo, "#999"), size=7, opacity=0.7, line=dict(color="white", width=0.4)),
                        text=sub["nombre"],
                        hovertemplate=f"<b>%{{text}}</b><br>Tipo: {tipo}<extra></extra>",
                    )
                )

        elif color_by == "Rating":
            fig_u.add_trace(
                go.Scatter(
                    x=df_f["umap_1"],
                    y=df_f["umap_2"],
                    mode="markers",
                    marker=dict(color=df_f["rating"], colorscale="Viridis", size=7, opacity=0.75, colorbar=dict(title="Rating"), showscale=True, line=dict(color="white", width=0.4)),
                    text=df_f["nombre"],
                    hovertemplate="<b>%{text}</b><br>Rating: %{marker.color:.1f}★<extra></extra>",
                )
            )

        else:
            fig_u.add_trace(
                go.Scatter(
                    x=df_f["umap_1"],
                    y=df_f["umap_2"],
                    mode="markers",
                    marker=dict(color=df_f["precio_num"], colorscale="Oranges", size=7, opacity=0.75, colorbar=dict(title="Precio Bs"), showscale=True, line=dict(color="white", width=0.4)),
                    text=df_f["nombre"],
                    hovertemplate="<b>%{text}</b><br>Precio estimado: Bs %{marker.color:.1f}<extra></extra>",
                )
            )

        fig_u.update_layout(
            title=f"UMAP — {color_by}",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            plot_bgcolor="#F8FAFC",
            paper_bgcolor="white",
            height=500,
            legend=dict(orientation="v", x=1.01, y=0.5),
        )
        st.plotly_chart(fig_u, use_container_width=True)

    with col_text:
        st.markdown("#### Lectura metodológica")
        st.markdown(
            """
            UMAP permite observar la estructura de similitud entre establecimientos.

            El color por **zona** muestra los clusters HDBSCAN.
            El color por **tipo** sirve para comprobar que las zonas no replican una etiqueta previa.
            """
        )

# ============================================================
# ANÁLISIS
# ============================================================
with t_analisis:
    st.markdown('<div class="section-hdr">Análisis de clusters</div>', unsafe_allow_html=True)
    df_p = df_f[df_f["cluster_hdbscan"] != -1].copy()

    if df_p.empty:
        st.warning("No hay datos para graficar con los filtros actuales.")
    else:
        df_p["zona"] = df_p["cluster_hdbscan"].map(lambda c: f"{CLUSTERS[c]['emoji']} {CLUSTERS[c]['nombre']}")
        cl_vals = sorted(df_p["cluster_hdbscan"].unique())
        color_map = {f"{CLUSTERS[c]['emoji']} {CLUSTERS[c]['nombre']}": CLUSTERS[c]["color"] for c in cl_vals}
        zona_order = list(color_map.keys())

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            fig_box = px.box(
                df_p,
                x="rating",
                y="zona",
                color="zona",
                color_discrete_map=color_map,
                title="Rating por zona",
                labels={"rating": "Rating", "zona": ""},
                category_orders={"zona": zona_order},
            )
            fig_box.update_layout(showlegend=False, height=380, plot_bgcolor="#F8FAFC")
            st.plotly_chart(fig_box, use_container_width=True)

        with r1c2:
            pf_dens = df_p.groupby("zona", as_index=False)["densidad_500m"].mean()
            fig_d = px.bar(
                pf_dens,
                x="zona",
                y="densidad_500m",
                color="zona",
                color_discrete_map=color_map,
                title="Densidad media en 500m",
                labels={"densidad_500m": "Densidad", "zona": ""},
                category_orders={"zona": zona_order},
            )
            fig_d.update_layout(showlegend=False, height=380, plot_bgcolor="#F8FAFC", xaxis_tickangle=-30)
            st.plotly_chart(fig_d, use_container_width=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            pf_res = df_p.groupby("zona", as_index=False)["n_resenas"].mean()
            fig_res = px.bar(
                pf_res,
                x="zona",
                y="n_resenas",
                color="zona",
                color_discrete_map=color_map,
                title="Popularidad: reseñas promedio",
                labels={"n_resenas": "Reseñas", "zona": ""},
                category_orders={"zona": zona_order},
            )
            fig_res.update_layout(showlegend=False, height=340, plot_bgcolor="#F8FAFC", xaxis_tickangle=-30)
            st.plotly_chart(fig_res, use_container_width=True)

        with r2c2:
            mix = df_p.groupby(["zona", "tipo_cat"]).size().reset_index(name="n")
            fig_mix = px.bar(
                mix,
                x="zona",
                y="n",
                color="tipo_cat",
                color_discrete_map=TIPO_COLOR,
                title="Mix de tipos por zona",
                labels={"n": "Lugares", "zona": "", "tipo_cat": "Tipo"},
                category_orders={"zona": zona_order},
            )
            fig_mix.update_layout(height=340, plot_bgcolor="#F8FAFC", xaxis_tickangle=-30, legend=dict(orientation="h", y=-0.4))
            st.plotly_chart(fig_mix, use_container_width=True)

        st.markdown("#### Heatmap de perfil normalizado")
        pf_agg = df_p.groupby("zona").agg(
            Rating=("rating", "mean"),
            Popularidad=("n_resenas", "mean"),
            Precio=("precio_num", "mean"),
            Densidad=("densidad_500m", "mean"),
        )
        hm = pd.DataFrame(MinMaxScaler().fit_transform(pf_agg), index=pf_agg.index, columns=pf_agg.columns)
        fig_hm = px.imshow(hm, text_auto=".2f", color_continuous_scale="Teal", aspect="auto", title="Perfil por zona — escala 0 a 1")
        fig_hm.update_layout(height=330)
        st.plotly_chart(fig_hm, use_container_width=True)

# ============================================================
# MODELO
# ============================================================
with t_modelos:
    st.markdown('<div class="section-hdr">Resumen del modelo</div>', unsafe_allow_html=True)

    resumen = pd.DataFrame(
        [
            {"Indicador": "Lugares totales", "Valor": len(df)},
            {"Indicador": "Clusters HDBSCAN", "Valor": n_clusters},
            {"Indicador": "Outliers", "Valor": f"{n_outliers} ({outlier_pct:.1f}%)"},
            {"Indicador": "Features conceptuales", "Valor": "lat, lon, densidad, rating, reseñas, precio"},
            {"Indicador": "Tipo de establecimiento", "Valor": "Solo interpretación post-hoc"},
        ]
    )
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.info(
        "HDBSCAN es adecuado porque no exige definir previamente el número de clusters y puede detectar puntos de ruido. "
        "En este proyecto, los tipos de establecimiento no se usan como input del modelo para evitar una clusterización trivial por categoría."
    )

# ============================================================
# DATOS
# ============================================================
with t_datos:
    st.markdown('<div class="section-hdr">Dataset filtrado</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        busq = st.text_input("🔍 Buscar por nombre", "")
    with d2:
        ord_cols = [c for c in ["rating", "n_resenas", "densidad_500m", "precio_num", "nombre"] if c in df_f.columns]
        ord_c = st.selectbox("Ordenar por", ord_cols)
    with d3:
        ord_d = st.radio("Orden", ["↓ Desc", "↑ Asc"], horizontal=True)

    df_t = df_f.copy()
    if busq:
        df_t = df_t[df_t["nombre"].str.contains(busq, case=False, na=False)]
    df_t = df_t.sort_values(ord_c, ascending=(ord_d == "↑ Asc"))

    st.markdown(f"**{len(df_t)} lugares**")
    cols_show = ["nombre", "tipo_cat", "cluster_nombre", "rating", "n_resenas", "precio_show", "densidad_500m", "direccion"]
    st.dataframe(
        df_t[cols_show].rename(
            columns={
                "nombre": "Nombre",
                "tipo_cat": "Tipo",
                "cluster_nombre": "Zona",
                "rating": "⭐",
                "n_resenas": "💬",
                "precio_show": "💰",
                "densidad_500m": "Densidad 500m",
                "direccion": "Dirección",
            }
        ),
        use_container_width=True,
        height=480,
    )
    st.download_button("⬇️ Descargar CSV filtrado", df_t.to_csv(index=False).encode("utf-8"), "lapaz_filtrado.csv", "text/csv")

# ============================================================
# INTERPRETACIÓN
# ============================================================
with t_interp:
    st.markdown('<div class="section-hdr">Interpretación automática de clusters</div>', unsafe_allow_html=True)
    st.markdown(
        """
        La interpretación se realiza después del clustering. El tipo de establecimiento se usa solo para describir
        la composición de cada zona, no para formar los clusters.
        """
    )

    for cl in cluster_ids_no_noise:
        info = CLUSTERS[cl]
        sub = df[df["cluster_hdbscan"] == cl]
        tipos_mix = sub["tipo_cat"].value_counts()
        top5 = sub.nlargest(5, "n_resenas")[["nombre", "tipo_cat", "rating", "n_resenas"]].reset_index(drop=True)
        top5.index += 1

        with st.expander(f"{info['emoji']} Cluster {cl} — {info['nombre']} · {len(sub)} lugares", expanded=(cl == cluster_ids_no_noise[0])):
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Rating medio", f"{sub['rating'].mean():.2f}★")
            i2.metric("Reseñas prom.", f"{int(sub['n_resenas'].mean())}")
            i3.metric("Precio prom.", f"Bs {sub['precio_num'].mean():.1f}")
            i4.metric("Densidad/500m", f"{sub['densidad_500m'].mean():.1f}")

            tipos_txt = ", ".join([f"{t} ({n})" for t, n in tipos_mix.head(4).items()])
            st.markdown(
                f"**Lectura del cluster:** Esta zona reúne {len(sub)} establecimientos con "
                f"rating medio de {sub['rating'].mean():.2f}, precio promedio de Bs {sub['precio_num'].mean():.1f}, "
                f"densidad media de {sub['densidad_500m'].mean():.1f} lugares en 500m y {sub['n_resenas'].mean():.0f} reseñas promedio. "
                f"Su composición post-hoc incluye principalmente: {tipos_txt}."
            )

            p1, p2 = st.columns([1, 1])
            with p1:
                fig_pie = px.pie(
                    values=tipos_mix.values,
                    names=tipos_mix.index,
                    color=tipos_mix.index,
                    color_discrete_map=TIPO_COLOR,
                    hole=0.35,
                )
                fig_pie.update_layout(height=230, margin=dict(t=0, b=0, l=0, r=0), legend=dict(font=dict(size=9)))
                st.plotly_chart(fig_pie, use_container_width=True)

            with p2:
                st.markdown("**Top 5 más reseñados:**")
                st.dataframe(
                    top5.rename(columns={"nombre": "Nombre", "tipo_cat": "Tipo", "rating": "⭐", "n_resenas": "💬"}),
                    use_container_width=True,
                    height=215,
                )

    st.success(
        "Conclusión: los clusters se interpretan como zonas urbanas con patrones similares de ubicación, densidad, rating, precio y popularidad. "
        "El tipo de establecimiento solo permite explicar la composición interna de cada zona."
    )
