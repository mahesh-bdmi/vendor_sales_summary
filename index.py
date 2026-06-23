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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def fmt_currency_col(series: pd.Series) -> pd.Series:
    """e.g. 50123456.0 -> '$50.12M' — for display tables, not raw exports."""
    return series.map(lambda v: f"${format_num(v)}")


def fmt_pct_col(series: pd.Series, decimals: int = 1) -> pd.Series:
    return series.map(lambda v: f"{v:.{decimals}f}%")


def fmt_ratio_col(series: pd.Series, decimals: int = 2) -> pd.Series:
    return series.map(lambda v: f"{v:.{decimals}f}")


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


def ensure_report_data() -> None:
    """Make sure report_ready_data.csv exists, rebuilding it from the
    database if needed. Shared by every page so the rebuild only happens
    once, no matter which page the user lands on first."""
    if REPORT_CSV.exists():
        return
    st.warning("Report data not found — building it now from the database.")
    if not SUMMARY_CSV.exists():
        if not DB_PATH.exists():
            st.error(f"Database not found at '{DB_PATH}'. Cannot build the report.")
            st.stop()
        summary_table_export()
    process_csv()
    load_report_data.clear()  # bust the cache since the file just changed


def get_vendor_brand_views(df: pd.DataFrame):
    """Shared vendor/brand aggregates used by both pages."""
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
    return ven, brand


def build_pareto_chart(top10: pd.DataFrame, grand_total: float, colors: dict):
    """Bar chart of purchase $ per vendor with a cumulative-% line on a
    secondary axis — the classic Pareto chart for the 80/20 view."""
    cum_pct = top10["TotalPurchaseDollars"].cumsum() / grand_total * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=top10["VendorName"],
            y=top10["TotalPurchaseDollars"],
            name="Purchase $",
            marker_color=PIE_PALETTE[0],
            marker_line_width=0,
            text=[f"${v:,.0f}" for v in top10["TotalPurchaseDollars"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=top10["VendorName"],
            y=cum_pct,
            name="Cumulative %",
            mode="lines+markers",
            line=dict(color=TARGET_YES_COLOR, width=2),
            marker=dict(size=7, color=TARGET_YES_COLOR),
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color=colors["threshold_color"],
        annotation_text="80%",
        annotation_font_size=10,
        annotation_font_color=colors["muted_text"],
        secondary_y=True,
    )
    fig.update_layout(
        title="<b>Pareto Chart — Top 10 Vendors by Purchase $</b>", bargap=0.25
    )
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(title_text="Purchase $", secondary_y=False)
    fig.update_yaxes(
        title_text="Cumulative %", secondary_y=True, range=[0, 110], ticksuffix="%"
    )
    return fig


def build_turnover_chart(low_turnover_df: pd.DataFrame, colors: dict):
    """Horizontal bar chart of the lowest-turnover vendors. Shared by the
    Dashboard and Analysis Summary pages so they stay visually consistent."""
    fig = px.bar(
        low_turnover_df.sort_values("StockTurnover", ascending=True),
        y="VendorName",
        x="StockTurnover",
        orientation="h",
        title=f"<b>Vendors with Low Turnover</b><br><span style='font-size:11px;color:{colors['muted_text']}'>Darker = bigger risk</span>",
        color="StockTurnover",
        color_continuous_scale=px.colors.sequential.Reds_r,
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_showscale=False)
    fig.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside",
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Turnover: %{x:.2f}<extra></extra>",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def build_margin_chart(brands_df: pd.DataFrame, colors: dict):
    """Horizontal bar chart of profit margin for a small set of brands —
    used for the hidden-gem (low sales, high margin) table on the summary page."""
    fig = px.bar(
        brands_df.sort_values("ProfitMargin", ascending=False),
        y="Description",
        x="ProfitMargin",
        orientation="h",
        title=f"<b>Profit Margin</b><br><span style='font-size:11px;color:{colors['muted_text']}'>Hidden-gem brands</span>",
        color="ProfitMargin",
        color_continuous_scale=px.colors.sequential.Purples,
    )
    fig.update_layout(xaxis_title=None, yaxis_title=None, coloraxis_showscale=False)
    fig.update_traces(
        texttemplate="%{x:.1f}%",
        textposition="outside",
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Margin: %{x:.1f}%<extra></extra>",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def render_table(df: pd.DataFrame, colors: dict) -> None:
    """Render a DataFrame as a static HTML table that's centered, sized to
    its content (not stretched full-width like st.dataframe), and themed
    to match the current light/dark mode."""
    styled = df.style.hide(axis="index").set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("margin", "0 auto"),
                    ("width", "auto"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("padding", "6px 18px"),
                    ("background-color", colors["hover_bg"]),
                    ("color", colors["font_color"]),
                    ("border", f"1px solid {colors['grid_color']}"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("padding", "6px 18px"),
                    ("color", colors["font_color"]),
                    ("border", f"1px solid {colors['grid_color']}"),
                ],
            },
        ]
    )
    st.markdown(styled.to_html(), unsafe_allow_html=True)


def dashboard_page() -> None:
    st.title("VENDOR SALES DASHBOARD")

    theme = get_theme_mode()
    colors = THEME_COLORS[theme]

    ensure_report_data()
    df = load_report_data()
    ven, brand = get_vendor_brand_views(df)

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

    fig_turnover = build_turnover_chart(
        ven.sort_values("StockTurnover", ascending=True).head(10), colors
    )
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


# ---------------------------------------------------------------------------
# Page 2: Written summary of the analysis (computed live from the data,
# so it never goes stale relative to whatever's in report_ready_data.csv)
# ---------------------------------------------------------------------------
def summary_page() -> None:
    st.title("VENDOR ANALYSIS SUMMARY")

    theme = get_theme_mode()
    colors = THEME_COLORS[theme]

    ensure_report_data()
    df = load_report_data()
    ven, brand = get_vendor_brand_views(df)

    total_sales = df["TotalSalesDollar"].sum()
    total_purchases = df["TotalPurchaseDollars"].sum()
    gross_profit = df["GrossProfit"].sum()
    avg_margin = df["ProfitMargin"].mean()
    unsold_value = (
        (df["TotalPurchaseQuantity"] - df["TotalSalesQuantity"]) * df["PurchasePrice"]
    ).sum()

    top10 = ven.sort_values("TotalPurchaseDollars", ascending=False).head(10)
    top10_pct = (
        top10["TotalPurchaseDollars"].sum() / ven["TotalPurchaseDollars"].sum() * 100
    )
    top_vendor = top10.iloc[0]

    top_brand = brand.sort_values("TotalSalesDollars", ascending=False).iloc[0]

    low_turnover = ven.sort_values("StockTurnover", ascending=True).head(10)

    low_sales_threshold = brand["TotalSalesDollars"].quantile(LOW_SALES_QUANTILE)
    high_margin_threshold = brand["ProfitMargin"].quantile(HIGH_MARGIN_QUANTILE)
    brand["TargetBrand"] = np.where(
        (brand["TotalSalesDollars"] < low_sales_threshold)
        & (brand["ProfitMargin"] > high_margin_threshold),
        "Yes",
        "No",
    )
    target_brands = brand[brand["TargetBrand"] == "Yes"].sort_values(
        "ProfitMargin", ascending=False
    )

    st.markdown(
        f"Across **{ven.shape[0]} vendors** and **{brand.shape[0]} brands**, this dataset represents "
        f"**\\${format_num(total_sales)}** in sales against **\\${format_num(total_purchases)}** in purchases — "
        f"a gross profit of **\\${format_num(gross_profit)}** at an average margin of **{avg_margin:.1f}%**. "
        f"**\\${format_num(unsold_value)}** of purchased inventory has not yet sold through."
    )

    st.divider()

    st.subheader("1. Vendor spend is highly concentrated")
    st.markdown(
        f"The top 10 vendors account for **{top10_pct:.1f}%** of all purchase dollars — a classic Pareto "
        f"distribution. **{top_vendor['VendorName']}** alone drives **{top_vendor['purchase_contribution']}** "
        f"of purchases (\\${format_num(top_vendor['TotalPurchaseDollars'])}). This concentration is useful "
        f"leverage in vendor negotiations, but also represents supply-chain risk if a top vendor is disrupted."
    )
    pareto_fig = build_pareto_chart(top10, ven["TotalPurchaseDollars"].sum(), colors)
    style_fig(pareto_fig, colors, height=440, show_legend=True)

    top10_display = top10[
        ["VendorName", "TotalPurchaseDollars", "purchase_contribution"]
    ].copy()
    top10_display["TotalPurchaseDollars"] = fmt_currency_col(
        top10_display["TotalPurchaseDollars"]
    )
    top10_display = top10_display.rename(
        columns={
            "TotalPurchaseDollars": "Purchase $",
            "purchase_contribution": "% of Total",
        }
    )

    chart_col, table_col = st.columns([3, 2])
    with chart_col, st.container(border=True):
        st.plotly_chart(pareto_fig, use_container_width=True)
    with table_col, st.container(border=True):
        render_table(top10_display, colors)

    st.subheader("2. Revenue is led by a small set of flagship brands")
    st.markdown(
        f"**{top_brand['Description']}** is the top-selling brand at **\\${format_num(top_brand['TotalSalesDollars'])}** "
        "in revenue. Brand-level revenue leaders tend to mirror the vendor rankings above — the largest "
        "vendors are largely winning on the back of a handful of marquee products."
    )

    st.subheader("3. A long tail of vendors shows weak inventory turnover")
    st.markdown(
        "These vendors are selling through the smallest share of what they purchase — capital and shelf "
        "space tied up with the least to show for it:"
    )
    turnover_fig = build_turnover_chart(low_turnover, colors)
    style_fig(turnover_fig, colors, height=420)

    low_turnover_display = low_turnover[
        ["VendorName", "StockTurnover", "TotalPurchaseDollars"]
    ].copy()
    low_turnover_display["StockTurnover"] = fmt_ratio_col(
        low_turnover_display["StockTurnover"]
    )
    low_turnover_display["TotalPurchaseDollars"] = fmt_currency_col(
        low_turnover_display["TotalPurchaseDollars"]
    )
    low_turnover_display = low_turnover_display.rename(
        columns={
            "StockTurnover": "Turnover Ratio",
            "TotalPurchaseDollars": "Purchase $",
        }
    )

    chart_col, table_col = st.columns([3, 2])
    with chart_col, st.container(border=True):
        st.plotly_chart(turnover_fig, use_container_width=True)
    with table_col, st.container(border=True):
        render_table(low_turnover_display, colors)

    st.subheader("4. Hidden-gem brands: low visibility, high margin")
    st.markdown(
        f"**{len(target_brands)} brands** have low total sales but profit margins above the "
        f"{HIGH_MARGIN_QUANTILE:.0%} threshold ({high_margin_threshold:.1f}%+) — strong candidates for a "
        "marketing push or better shelf placement, since the margin is already proven and the bottleneck "
        "is simply volume."
    )
    target_brands_top15 = target_brands.head(15)
    margin_fig = build_margin_chart(target_brands_top15, colors)
    style_fig(margin_fig, colors, height=440)

    target_brands_display = target_brands_top15[
        ["Description", "TotalSalesDollars", "ProfitMargin"]
    ].copy()
    target_brands_display["TotalSalesDollars"] = fmt_currency_col(
        target_brands_display["TotalSalesDollars"]
    )
    target_brands_display["ProfitMargin"] = fmt_pct_col(
        target_brands_display["ProfitMargin"]
    )
    target_brands_display = target_brands_display.rename(
        columns={"TotalSalesDollars": "Sales $", "ProfitMargin": "Margin %"}
    )

    chart_col, table_col = st.columns([3, 2])
    with chart_col, st.container(border=True):
        st.plotly_chart(margin_fig, use_container_width=True)
    with table_col, st.container(border=True):
        render_table(target_brands_display, colors)


if __name__ == "__main__":
    st.set_page_config(page_title="Vendor Sales Dashboard", layout="wide")
    nav = st.navigation(
        [
            st.Page(dashboard_page, title="Dashboard", icon="📊", default=True),
            st.Page(summary_page, title="Analysis Summary", icon="📝"),
        ]
    )
    nav.run()
