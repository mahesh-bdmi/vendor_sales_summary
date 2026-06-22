"""
Vendor Sales Dashboard
-----------------------
Pipeline: SQLite extraction -> feature engineering -> Streamlit/Plotly report.

Run as a script to (re)build the CSVs, or via `streamlit run index.py` to
view the dashboard. The dashboard will auto-build the CSVs if they don't
exist yet, so the three stages no longer have to be run manually in order.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import sqlite3
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path("inventory.db")
SUMMARY_CSV = Path("vendor_sales_summary.csv")
FREIGHT_CSV = Path("freight_summary.csv")
REPORT_CSV = Path("report_ready_data.csv")

LOW_SALES_QUANTILE = 0.15
HIGH_MARGIN_QUANTILE = 0.85
SCATTER_SALES_CAP = 5000  # cuts off outliers for readability in the scatter plot

# ---------------------------------------------------------------------------
# Chart theme — one shared palette + layout so every chart looks like part
# of the same dashboard instead of plotly's defaults. Colors are picked
# per-mode so the charts stay readable on both Streamlit's light and dark
# themes (transparent backgrounds mean plotly text/lines must match
# whatever's behind them).
# ---------------------------------------------------------------------------
PIE_PALETTE = [
    "#5B8FF9",
    "#5AD8A6",
    "#F6BD16",
    "#E8684A",
    "#6DC8EC",
    "#9270CA",
    "#FF9D4D",
    "#269A99",
    "#FF99C3",
    "#5D7092",
]
TARGET_YES_COLOR = "#E8684A"

THEME_COLORS = {
    "light": dict(
        font_color="#374151",
        title_color="#111827",
        grid_color="#EEF0F3",
        hover_bg="white",
        muted_text="#6B7280",
        threshold_color="#9CA3AF",
        no_color="#CBD5E1",
        marker_border="white",
    ),
    "dark": dict(
        font_color="#E5E7EB",
        title_color="#F9FAFB",
        grid_color="#374151",
        hover_bg="#1F2937",
        muted_text="#9CA3AF",
        threshold_color="#6B7280",
        no_color="#4B5563",
        marker_border="#0E1117",  # Streamlit's default dark background
    ),
}


def get_theme_mode() -> str:
    """Detect whether Streamlit is running in light or dark mode so charts
    can match it. Falls back to 'dark' if detection isn't available."""
    try:
        theme_type = st.context.theme.type  # Streamlit >= 1.36
        if theme_type in THEME_COLORS:
            return theme_type
    except Exception:
        pass
    base = st.get_option("theme.base")
    return base if base in THEME_COLORS else "dark"


def style_fig(fig, colors: dict, height: int = 380, show_legend: bool = False):
    """Apply a consistent theme (font, spacing, gridlines, transparency)
    to every chart in the dashboard."""
    fig.update_layout(
        font=dict(family="Arial, sans-serif", size=13, color=colors["font_color"]),
        title_font=dict(size=16, color=colors["title_color"]),
        title_x=0.02,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=30, t=60, b=10),
        height=height,
        showlegend=show_legend,
        hoverlabel=dict(
            bgcolor=colors["hover_bg"],
            font_size=12,
            font_family="Arial",
            font_color=colors["font_color"],
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor=colors["grid_color"], zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: Extract + join raw tables into a summary table
# ---------------------------------------------------------------------------
def summary_table_export(db_path: Path = DB_PATH) -> None:
    """Connect to the database, build the vendor/brand summary table and the
    freight summary, and export both to CSV."""

    logger.info("Building the final summary data")

    with sqlite3.connect(db_path) as conn:
        freight_summary = pd.read_sql_query(
            """
            SELECT VendorNumber, VendorName, SUM(Freight) AS FreightCost
            FROM vendor_invoice
            GROUP BY VendorNumber, VendorName
            """,
            conn,
        )

        df = pd.read_sql_query(
            """
            WITH sales_summary AS (
                SELECT
                    VendorNo, VendorName, Brand, SalesPrice,
                    SUM(SalesQuantity) AS TotalSalesQuantity,
                    SUM(SalesDollars) AS TotalSalesDollar,
                    SUM(ExciseTax) AS TotalExciseTax
                FROM sales
                GROUP BY VendorNo, Brand
            ),
            purchase_summary AS (
                SELECT
                    pp.VendorNumber, p.VendorName, pp.Brand,
                    pp.Price AS ActualPrice, p.PurchasePrice, pp.Volume, pp.Description,
                    SUM(p.Quantity) AS TotalPurchaseQuantity,
                    SUM(p.Dollars) AS TotalPurchaseDollars
                FROM purchases p
                JOIN purchase_prices pp ON pp.Brand = p.Brand
                WHERE p.PurchasePrice > 0
                GROUP BY pp.VendorNumber, p.VendorName, pp.Brand, pp.Price,
                         p.PurchasePrice, pp.Volume, pp.Description
            )
            SELECT
                ps.VendorNumber, ps.VendorName, ps.Brand, ps.Description,
                ps.ActualPrice, ps.PurchasePrice, ss.SalesPrice, ps.Volume,
                ps.TotalPurchaseQuantity, ps.TotalPurchaseDollars,
                ss.TotalSalesQuantity, ss.TotalSalesDollar, ss.TotalExciseTax
            FROM purchase_summary ps
            LEFT JOIN sales_summary ss
                ON ss.VendorNo = ps.VendorNumber AND ss.Brand = ps.Brand
            ORDER BY ps.TotalPurchaseDollars DESC
            """,
            conn,
        )

    df.to_csv(SUMMARY_CSV, index=False)
    freight_summary.to_csv(FREIGHT_CSV, index=False)
    logger.info("Exported %s and %s", SUMMARY_CSV, FREIGHT_CSV)


# ---------------------------------------------------------------------------
# Stage 2: Feature engineering on the summary table
# ---------------------------------------------------------------------------
def process_csv() -> None:
    """Add derived metrics (margin, turnover, etc.) and export the
    report-ready CSV used by the dashboard."""

    df = pd.read_csv(SUMMARY_CSV)

    df.fillna(0, inplace=True)  # fillna BEFORE casting, or NaNs break astype("int64")
    df["Volume"] = df["Volume"].astype("int64")
    df["VendorName"] = df["VendorName"].str.strip()

    df["GrossProfit"] = df["TotalSalesDollar"] - df["TotalPurchaseDollars"]

    # Guard against divide-by-zero -> inf/NaN in the ratio columns.
    df["ProfitMargin"] = np.where(
        df["TotalSalesDollar"] != 0,
        (df["GrossProfit"] / df["TotalSalesDollar"]) * 100,
        0,
    )
    df["StockTurnover"] = np.where(
        df["TotalPurchaseQuantity"] != 0,
        df["TotalSalesQuantity"] / df["TotalPurchaseQuantity"],
        0,
    )
    df["SalesPurchaseRatio"] = np.where(
        df["TotalPurchaseDollars"] != 0,
        df["TotalSalesDollar"] / df["TotalPurchaseDollars"],
        0,
    )

    df.to_csv(REPORT_CSV, index=False)
    logger.info("Exported %s", REPORT_CSV)


# ---------------------------------------------------------------------------
# Stage 3: Dashboard
# ---------------------------------------------------------------------------
def format_num(x: float) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.2f}K"
    return f"{x:.0f}"


@st.cache_data
def load_report_data() -> pd.DataFrame:
    df = pd.read_csv(REPORT_CSV)
    df = df[
        (df["GrossProfit"] > 0)
        & (df["ProfitMargin"] > 0)
        & (df["TotalSalesQuantity"] > 0)
    ]
    return df


def aggregate_by(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Shared groupby/aggregation logic used for both the vendor and brand views."""
    agg = (
        df.groupby(group_col)
        .agg(
            TotalPurchaseQuantity=("TotalPurchaseQuantity", "sum"),
            TotalPurchaseDollars=("TotalPurchaseDollars", "sum"),
            TotalSalesQuantity=("TotalSalesQuantity", "sum"),
            TotalSalesDollars=("TotalSalesDollar", "sum"),
            ProfitMargin=("ProfitMargin", "mean"),
        )
        .reset_index()
    )
    return agg


def build_report() -> None:
    st.set_page_config(page_title="Vendor Sales Dashboard", layout="wide")
    st.title("VENDOR SALES DASHBOARD")

    theme = get_theme_mode()
    colors = THEME_COLORS[theme]

    if not REPORT_CSV.exists():
        st.warning("Report data not found — building it now from the database.")
        if not SUMMARY_CSV.exists():
            if not DB_PATH.exists():
                st.error(f"Database not found at '{DB_PATH}'. Cannot build the report.")
                st.stop()
            summary_table_export()
        process_csv()
        load_report_data.clear()  # bust the cache since the file just changed

    df = load_report_data()

    ven = aggregate_by(df, "VendorName")
    ven["purchase_contribution"] = (
        ven["TotalPurchaseDollars"] / ven["TotalPurchaseDollars"].sum()
    ).map("{:.1%}".format)
    ven["StockTurnover"] = np.where(
        ven["TotalPurchaseQuantity"] != 0,
        ven["TotalSalesQuantity"] / ven["TotalPurchaseQuantity"],
        0,
    )

    brand = aggregate_by(df, "Description")

    # ---- KPI row ----
    col1, col2, col3, col4, col5 = st.columns(
        5, vertical_alignment="center", border=True
    )
    unsold_value = (
        (df["TotalPurchaseQuantity"] - df["TotalSalesQuantity"]) * df["PurchasePrice"]
    ).sum()

    col1.metric("Total Sales", f"${format_num(df['TotalSalesDollar'].sum())}")
    col2.metric("Total Purchases", f"${format_num(df['TotalPurchaseDollars'].sum())}")
    col3.metric("Gross Profit", f"${format_num(df['GrossProfit'].sum())}")
    col4.metric("Profit Margin (%)", f"{df['ProfitMargin'].mean():,.2f}")
    col5.metric("Unsold Value", f"${format_num(unsold_value)}")

    # ---- Vendor purchase contribution ----
    top10 = ven.sort_values("TotalPurchaseDollars", ascending=False).head(10)
    top10_pct = (
        top10["TotalPurchaseDollars"].sum() / ven["TotalPurchaseDollars"].sum() * 100
    )

    fig_pie = px.pie(
        top10,
        values="TotalPurchaseDollars",
        names="VendorName",
        title=f"<b>Vendor Purchase Contribution</b><br><span style='font-size:11px;color:{colors['muted_text']}'>Top 10 vendors</span>",
        hole=0.6,
        color_discrete_sequence=PIE_PALETTE,
    )
    fig_pie.update_traces(
        textinfo="percent",
        textposition="inside",
        marker=dict(line=dict(color=colors["marker_border"], width=2)),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>",
    )
    fig_pie.add_annotation(
        text=f"<b>{top10_pct:.1f}%</b><br><span style='font-size:11px;color:{colors['muted_text']}'>of Total</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=22, color=colors["title_color"]),
    )
    fig_pie.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )
    style_fig(fig_pie, colors, height=420, show_legend=True)

    fig_bar = px.bar(
        ven.sort_values("TotalSalesDollars", ascending=False).head(10),
        y="VendorName",
        x="TotalSalesDollars",
        orientation="h",
        title="<b>Top 10 Vendors by Revenue</b>",
        color="TotalSalesDollars",
        color_continuous_scale=px.colors.sequential.Blues,
    )
    fig_bar.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_showscale=False)
    fig_bar.update_traces(
        texttemplate="$%{x:.2s}",
        textposition="outside",
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    )
    fig_bar.update_yaxes(autorange="reversed")
    style_fig(fig_bar, colors, height=420)

    fig_bar_brand = px.bar(
        brand.sort_values("TotalSalesDollars", ascending=False).head(10),
        y="Description",
        x="TotalSalesDollars",
        orientation="h",
        title="<b>Top 10 Brands by Revenue</b>",
        color="TotalSalesDollars",
        color_continuous_scale=px.colors.sequential.Greens,
    )
    fig_bar_brand.update_layout(
        xaxis_title=None, yaxis_title=None, coloraxis_showscale=False
    )
    fig_bar_brand.update_traces(
        texttemplate="$%{x:.2s}",
        textposition="outside",
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    )
    fig_bar_brand.update_yaxes(autorange="reversed")
    style_fig(fig_bar_brand, colors, height=420)

    fig_turnover = px.bar(
        ven.sort_values("StockTurnover", ascending=True).head(10),
        y="VendorName",
        x="StockTurnover",
        orientation="h",
        title=f"<b>Vendors with Low Turnover</b><br><span style='font-size:11px;color:{colors['muted_text']}'>Darker = bigger risk</span>",
        color="StockTurnover",
        color_continuous_scale=px.colors.sequential.Reds_r,
    )
    fig_turnover.update_layout(
        xaxis_title=None, yaxis_title=None, coloraxis_showscale=False
    )
    fig_turnover.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside",
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Turnover: %{x:.2f}<extra></extra>",
    )
    fig_turnover.update_yaxes(autorange="reversed")
    style_fig(fig_turnover, colors, height=420)

    col11, col12, col13 = st.columns(3)
    for col, fig in zip((col11, col12, col13), (fig_pie, fig_bar, fig_bar_brand)):
        with col, st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

    # ---- Brands to target: low sales, high margin ----
    low_sales_threshold = brand["TotalSalesDollars"].quantile(LOW_SALES_QUANTILE)
    high_margin_threshold = brand["ProfitMargin"].quantile(HIGH_MARGIN_QUANTILE)
    brand["TargetBrand"] = np.where(
        (brand["TotalSalesDollars"] < low_sales_threshold)
        & (brand["ProfitMargin"] > high_margin_threshold),
        "Yes",
        "No",
    )
    brand_performance = brand[brand["TotalSalesDollars"] < SCATTER_SALES_CAP]

    target_brand_chart = px.scatter(
        brand_performance,
        x="TotalSalesDollars",
        y="ProfitMargin",
        hover_data=["Description"],
        color="TargetBrand",
        color_discrete_map={"Yes": TARGET_YES_COLOR, "No": colors["no_color"]},
        title="<b>Brands with Low Sales & High Returns</b>",
        opacity=0.85,
    )
    target_brand_chart.update_traces(
        marker=dict(size=10, line=dict(width=1, color=colors["marker_border"])),
        hovertemplate="<b>%{customdata[0]}</b><br>Sales: $%{x:,.0f}<br>Margin: %{y:.1f}%<extra></extra>",
    )
    target_brand_chart.add_vline(
        x=low_sales_threshold,
        line_dash="dash",
        line_color=colors["threshold_color"],
        annotation_text="Low-sales line",
        annotation_font_size=10,
        annotation_font_color=colors["muted_text"],
    )
    target_brand_chart.add_hline(
        y=high_margin_threshold,
        line_dash="dash",
        line_color=colors["threshold_color"],
        annotation_text="High-margin line",
        annotation_font_size=10,
        annotation_font_color=colors["muted_text"],
    )
    target_brand_chart.update_layout(
        xaxis_title="Total Sales ($)",
        yaxis_title="Profit Margin (%)",
        legend_title_text="Target brand",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    style_fig(target_brand_chart, colors, height=420, show_legend=True)

    left, right = st.columns([1, 2])
    with left, st.container(border=True):
        st.plotly_chart(fig_turnover, use_container_width=True)
    with right, st.container(border=True):
        st.plotly_chart(target_brand_chart, use_container_width=True)


if __name__ == "__main__":
    build_report()
