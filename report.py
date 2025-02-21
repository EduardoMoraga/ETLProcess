import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from supabase import create_client, Client
from mlxtend.frequent_patterns import apriori, association_rules

# 🔹 CONFIGURAR TU CONEXIÓN A SUPABASE
url = "https://wvzsqpzbbltpfuaeoaem.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind2enNxcHpiYmx0cGZ1YWVvYWVtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAwNTk3ODksImV4cCI6MjA1NTYzNTc4OX0.KDwcTkxs2mZVB6JIFOuP0rNjVW163gWvS1ECCjLtxDo"
supabase: Client = create_client(url, key)

# 🔹 FUNCIONES AUXILIARES
def obtener_datos(tabla):
    response = supabase.table(tabla).select("*").execute()
    return pd.DataFrame(response.data)

# 🔹 OBTENER DATOS
df_transacciones = obtener_datos("transacciones")
df_productos = obtener_datos("productos")
df_tiendas = obtener_datos("tiendas")

# 🔹 RENOMBRAR COLUMNAS PARA EVITAR ERRORES
df_tiendas.rename(columns={"id": "tienda_id", "nombre": "Tienda"}, inplace=True)
df_productos.rename(columns={"id": "producto_id", "nombre": "Producto"}, inplace=True)
df_transacciones["fecha"] = pd.to_datetime(df_transacciones["fecha"], errors="coerce")

# 🔹 UNIR DATOS
df_transacciones = df_transacciones.merge(df_productos, on="producto_id", how="left")
df_transacciones = df_transacciones.merge(df_tiendas, on="tienda_id", how="left")

# ======================= 🎨 INTERFAZ =======================
st.set_page_config(layout="wide", page_title="BI Retail - Análisis Estratégico")

st.title("📊 **Dashboard Estratégico de Retail**")
st.sidebar.header("⚙️ **Filtros de Análisis**")

# 🔹 FILTROS
rango_fechas = st.sidebar.date_input("📅 Rango de Fechas", [df_transacciones["fecha"].min(), df_transacciones["fecha"].max()])
tienda_select = st.sidebar.multiselect("🏪 Filtrar por Tienda", df_tiendas["Tienda"].unique(), default=df_tiendas["Tienda"].unique())
categoria_select = st.sidebar.multiselect("📦 Filtrar por Categoría", df_productos["categoria"].unique(), default=df_productos["categoria"].unique())
horizonte = st.sidebar.selectbox("📌 Inventario Inmovilizado (Días sin Venta)", [7, 14, 21])

# 🔹 APLICAR FILTROS
df_filtrado = df_transacciones[
    (df_transacciones["fecha"] >= pd.to_datetime(rango_fechas[0])) & 
    (df_transacciones["fecha"] <= pd.to_datetime(rango_fechas[1])) & 
    (df_transacciones["Tienda"].isin(tienda_select)) & 
    (df_transacciones["categoria"].isin(categoria_select))
]

# 🔹 KPIs CLAVES
col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total Ventas", f"${df_filtrado['precio_unitario'].sum():,.0f}")
col2.metric("🔄 Tickets Generados", f"{df_filtrado['cliente_id'].nunique()}")
col3.metric("📉 Productos en Riesgo", f"{df_filtrado[df_filtrado['cantidad'] < 5]['Producto'].nunique()}")
col4.metric("📊 Tiendas Activas", f"{df_filtrado['Tienda'].nunique()}")

# 🔹 ANÁLISIS DETALLADO DE PRODUCTOS
st.subheader("🔎 **Análisis Detallado de Productos**")
df_analisis = df_filtrado.groupby("Producto").agg({
    "precio_unitario": "mean",
    "cantidad": "sum"
}).reset_index()
df_analisis["DOS"] = df_analisis["cantidad"].apply(lambda x: x / df_analisis["cantidad"].mean())
df_analisis["Inventario_Valorizado"] = df_analisis["precio_unitario"] * df_analisis["cantidad"]
df_analisis["Pareto"] = pd.qcut(df_analisis["cantidad"].rank(method="first"), q=10, labels=[f"Decil {i}" for i in range(1, 11)])

st.dataframe(df_analisis)

# 🔹 CLASIFICACIÓN DE TIENDAS
st.subheader("🏪 **Clasificación de Tiendas**")
df_tiendas["Ventas_Totales"] = df_filtrado.groupby("Tienda")["precio_unitario"].sum().reset_index()["precio_unitario"]
df_tiendas["Inventario_Inmovilizado_%"] = (df_filtrado["cantidad"] / df_filtrado["cantidad"].sum()) * 100

st.dataframe(df_tiendas)

# 🔹 MARKET BASKET ANALYSIS
st.subheader("🛒 **Relación entre Productos Comprados Juntos**")
df_cestas = df_filtrado.groupby(["cliente_id", "Producto"])["cantidad"].sum().unstack().fillna(0)
df_cestas_bin = df_cestas.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(df_cestas_bin, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

rules["Regla"] = rules.apply(lambda x: f"Si compras {list(x['antecedents'])} hay {round(x['lift'],2)}x más chance de comprar {list(x['consequents'])}", axis=1)
st.dataframe(rules[["Regla", "support", "confidence"]])

# 🔹 RECOMENDACIONES
st.subheader("⚡ **Recomendaciones Estratégicas**")
top_traccionador = rules.sort_values("lift", ascending=False).iloc[0]["antecedents"]

st.markdown(f"""
📌 **Producto que más tracciona otras compras:** {top_traccionador}  
📌 **Optimizar precios en productos con alta elasticidad.**  
📌 **Reforzar stock en tiendas con alta demanda y baja disponibilidad.**  
📌 **Mejorar surtido en tiendas con baja diversidad de productos vendidos.**  
📌 **Implementar promociones en productos con baja rotación.**  
""")

st.success("✅ **Dashboard BI de Retail Cargado Correctamente!** 🚀")