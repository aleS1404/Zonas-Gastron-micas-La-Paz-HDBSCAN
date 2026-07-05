# Zonas Gastronómicas de La Paz — Proyecto Completo v2
**Aprendizaje No Supervisado · HDBSCAN + UMAP · Streamlit**

---

## Corrección aplicada en esta versión

En la versión anterior se incluía el **one-hot del tipo de establecimiento**
como feature del modelo. Eso hacía que HDBSCAN agrupara trivialmente por
categoría (todos los cafés juntos, todos los bares juntos), lo que equivale
a replicar una etiqueta que ya existía — contrario al espíritu del
aprendizaje no supervisado.

**En esta versión:**
- Features del modelo: `lat`, `lon`, `rating_norm`, `log_resenas`, `precio_num`, `densidad_500m`
- El tipo de establecimiento solo se usa **después** para interpretar los clusters
- Cada cluster tiene un **mix real de tipos**, lo que confirma que el modelo
  descubrió patrones geoespaciales genuinos

---

## Estructura del proyecto

```
proyecto_final/
├── app.py                        ← App Streamlit corregida
├── dataset_lapaz_clustered.csv   ← Dataset con clusters v2
├── zonas_gastronomicas_lapaz.ipynb ← Notebook corregido
├── requirements.txt              ← Dependencias
└── README.md                     ← Esta guía
```

---

## Resultados del modelo (v2)

| Cluster | Zona | n | Rating | Reseñas | Precio | Densidad |
|---|---|---|---|---|---|---|
| 5 | 🏙️ Núcleo Gastronómico Central | 102 | 4.25 | 616 | 1.89 | 61.4 |
| 1 | 🍽️ Zona Gastronómica Consolidada | 84 | 4.38 | 472 | 2.12 | 41.4 |
| 4 | 🔥 Zona de Alta Popularidad | 14 | 4.09 | 529 | 1.43 | 16.5 |
| 3 | 🛒 Mercados Populares | 23 | 3.58 | 305 | 0.00 | 4.4 |
| 2 | ⭐ Zona de Nicho | 12 | 4.80 | 11 | 0.25 | 15.2 |
| 0 | 📍 Zona Periférica | 5 | 3.80 | 169 | 0.40 | 5.8 |
| -1 | Outliers | 187 | — | — | — | — |

**Métricas HDBSCAN:** Silhouette=0.407 · Calinski-H.=138.1 · Ruido=43.8%

---

## Cómo levantar el proyecto

### 1. Requisitos previos
- Python 3.9 o superior instalado
- pip actualizado: `pip install --upgrade pip`

### 2. Crear entorno virtual (recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> ⚠️ `hdbscan` compila C++ — puede tardar 2-5 minutos.
> Si falla en Windows: `pip install wheel` primero, luego reintentar.
> Alternativa: `pip install --only-binary :all: hdbscan`

### 4. Verificar el dataset

`dataset_lapaz_clustered.csv` debe estar en la misma carpeta que `app.py`.
Es el output corregido del notebook (sin one-hot de tipo).

### 5. Levantar la app

```bash
streamlit run app.py
```

Se abre en: **http://localhost:8501**

Para cambiar puerto: `streamlit run app.py --server.port 8502`

---

## Tabs de la aplicación

| Tab | Contenido |
|---|---|
| 🗺️ **Mapa** | Folium interactivo con marcadores coloreados, popup con detalles, toggle de capas por zona |
| 🔬 **UMAP** | Scatter 2D coloreable por zona, tipo o rating |
| 📊 **Análisis** | Boxplots de rating, densidad, popularidad, **mix de tipos por zona** (barras apiladas) + heatmap |
| ⚖️ **Modelos** | Comparación KMeans vs DBSCAN vs HDBSCAN con métricas |
| 📋 **Datos** | Tabla filtrable con búsqueda y descarga CSV |
| 🧠 **Interpretación** | Análisis de cada cluster con pies de mix de tipos, top 5 más populares |

---

## Errores comunes

| Error | Solución |
|---|---|
| `ModuleNotFoundError: streamlit_folium` | `pip install streamlit-folium` |
| `hdbscan` falla en Windows | `pip install --only-binary :all: hdbscan` |
| `FileNotFoundError: dataset_lapaz_clustered.csv` | Ejecutar desde la carpeta del proyecto |
| Mapa en blanco | Revisar conexión a internet (tiles de CartoDB) |
| Puerto en uso | `streamlit run app.py --server.port 8502` |
