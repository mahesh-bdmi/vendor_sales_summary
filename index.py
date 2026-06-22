import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
from datetime import datetime
import streamlit as st
import plotly.express as px

# engine = create_engine("sqlite:///inventory.db")


# This functions connects to a database and builds a summary table by joining the different tables in the database
def summary_table_export():

    conn = sqlite3.connect("inventory.db")

    freight_summary = pd.read_sql_query(
        """select VendorNumber,VendorName, SUM(Freight) as FreightCost FROM vendor_invoice group by VendorNumber,VendorName""",
        conn,
    )

    print("Building the final summary data")
    start_time = datetime.now()
    print(f"Task started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Processing...")
    df = pd.read_sql_query(
        """
                                    WITH sales_summary AS
                                    (
                                    SELECT 
                                    VendorNo,VendorName,Brand,SalesPrice,
                                        SUM(SalesQuantity) AS TotalSalesQuantity,
                                        SUM(SalesDollars) AS TotalSalesDollar,
                                        SUM(ExciseTax) AS TotalExciseTax
                                    FROM sales
                                    GROUP BY VendorNo,Brand),
                                    
                                    purchase_summary AS
                                    (
                                    SELECT
                                        pp.VendorNumber,p.VendorName,pp.Brand,pp.Price as ActualPrice,p.PurchasePrice,pp.Volume,pp.Description,
                                        SUM(p.Quantity) AS TotalPurchaseQuantity,
                                        SUM(p.Dollars) AS  TotalPurchaseDollars
                                    FROM purchases p
                                    JOIN purchase_prices pp
                                    ON pp.Brand = p.Brand
                                    WHERE p.PurchasePrice > 0
                                    GROUP BY pp.VendorNumber,p.VendorName,pp.Brand,pp.Price,pp.PurchasePrice,pp.Volume,pp.Description
                                    )

                                    SELECT  ps.VendorNumber,ps.VendorName,ps.Brand,ps.Description,ps.ActualPrice,ps.PurchasePrice,ss.SalesPrice,Volume,
                                            ps.TotalPurchaseQuantity,ps.TotalPurchaseDollars,
                                            ss.TotalSalesQuantity,ss.TotalSalesDollar,ss.TotalExciseTax
                                    FROM sales_summary ss
                                    RIGHT JOIN purchase_summary ps
                                    ON ss.VendorNo = ps.VendorNumber AND ss.Brand = ps.Brand
                                ORDER BY TotalPurchaseDollars DESC
    """,
        conn,
    )
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(
        f"Final summary table has been created at {end_time.strftime("%Y-%m-%d %H:%M:%S")}"
    )
    print(f"That took {elapsed_time} seconds")
    print("Exporting the data now....")
    df.to_csv("vendor_sales_summary.csv", index=False)
    freight_summary.to_csv("freight_summary.csv", index=False)
    print("Files are created")


# This functions adds the requited addoitonal fields to the sumamry table and exports it to a file
def process_csv():

    df = pd.read_csv("vendor_sales_summary.csv")

    df["Volume"] = df["Volume"].astype("int64")
    df.fillna(0, inplace=True)
    df["VendorName"] = df["VendorName"].str.strip()
    df["GrossProfit"] = df["TotalSalesDollar"] - df["TotalPurchaseDollars"]
    df["ProfitMargin"] = (df["GrossProfit"] / df["TotalSalesDollar"]) * 100
    df["StockTurnover"] = df["TotalSalesQuantity"] / df["TotalPurchaseQuantity"]
    df["SalesPurchaseRatio"] = df["TotalSalesDollar"] / df["TotalPurchaseDollars"]
    df.rename(columns={"TotalSalesDollar": "TotalSalesDollars"})

    df.to_csv("report_ready_data.csv", index=False)


# This function builds the required graphs by using the Streamlit and Plotly python libraries
def build_report():

    format_num = lambda x: (
        f"{x/1_000_000:.2f}M"
        if abs(x) >= 1_000_000
        else f"{x/1_000:.2f}K" if abs(x) >= 1_000 else f"{x:.0f}"
    )
    st.set_page_config(page_title="Vendor Sales Dashboard", layout="wide")
    st.title("Vendor Sales Dashboard".upper())

    df = pd.read_csv("report_ready_data.csv")

    df = df[df["GrossProfit"] > 0]
    df = df[df["ProfitMargin"] > 0]
    df = df[df["TotalSalesQuantity"] > 0]

    ven = (
        df.groupby("VendorName")
        .agg(
            {
                "TotalPurchaseQuantity": "sum",
                "TotalPurchaseDollars": "sum",
                "TotalSalesQuantity": "sum",
                "TotalSalesDollar": "sum",
                "ProfitMargin": "mean",
            }
        )
        .reset_index()
    )
    ven["purchase_contribution"] = (
        ven["TotalPurchaseDollars"] / ven["TotalPurchaseDollars"].sum()
    ).map("{:.1%}".format)

    ven["StockTurnover"] = ven["TotalSalesQuantity"] / ven["TotalPurchaseQuantity"]

    brand = (
        df.groupby("Description")
        .agg(
            {
                "TotalPurchaseQuantity": "sum",
                "TotalPurchaseDollars": "sum",
                "TotalSalesQuantity": "sum",
                "TotalSalesDollar": "sum",
                "ProfitMargin": "mean",
            }
        )
        .reset_index()
    )

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(
        5, vertical_alignment="center", border=True
    )
    col1.metric(
        "Total Sales",
        f"${format_num(df['TotalSalesDollar'].sum())}",
    )
    col2.metric("Total Purchases", f"${format_num(df['TotalPurchaseDollars'].sum())}")
    col3.metric("Gross Profit", f"${format_num(df['GrossProfit'].sum())}")
    col4.metric("Profit Margin (%)", f"{df['ProfitMargin'].mean():,.2f}")
    col5.metric(
        "Unsold Value",
        f"${format_num(((df["TotalPurchaseQuantity"] - df["TotalSalesQuantity"]) * df["PurchasePrice"]).sum())}",
    )

    # Vendor purchases contribution
    top10 = ven.sort_values("TotalPurchaseDollars", ascending=False).head(10)
    top10_pct = (
        top10["TotalPurchaseDollars"].sum() / ven["TotalPurchaseDollars"].sum() * 100
    )

    fig_pie = px.pie(
        top10,
        values="TotalPurchaseDollars",
        names="VendorName",
        title="Vendor Purchase Contribution (Top 10)",
        hole=0.3,
    )
    fig_pie.add_annotation(
        text=f"{top10_pct:.1f}%<br> of Total",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15),
    )
    # Top 10 Vendors by Sales
    fig_bar = px.bar(
        ven.sort_values("TotalSalesDollar", ascending=False).head(10).reset_index(),
        y="VendorName",
        x="TotalSalesDollar",
        orientation="h",
        title="Top 10 Vendors by Revenue",
    )
    fig_bar.update_layout(xaxis_title=None, yaxis_title=None)
    fig_bar.update_traces(texttemplate="%{value:.2s}", textposition="outside")
    fig_bar.update_yaxes(autorange="reversed")

    # Top 10 Brands by Sales
    fig_bar_brand = px.bar(
        brand.sort_values("TotalSalesDollar", ascending=False).head(10).reset_index(),
        y="Description",
        x="TotalSalesDollar",
        orientation="h",
        title="Top 10 Brands by Revenue",
    )
    fig_bar_brand.update_layout(xaxis_title=None, yaxis_title=None)
    fig_bar_brand.update_traces(texttemplate="%{value:.2s}", textposition="outside")
    fig_bar_brand.update_yaxes(autorange="reversed")

    fig_turnover = px.bar(
        ven.sort_values("StockTurnover", ascending=True).head(10).reset_index(),
        y="VendorName",
        x="StockTurnover",
        orientation="h",
        title="Vendors with Low Turnover",
    )
    fig_turnover.update_layout(xaxis_title=None, yaxis_title=None)
    fig_turnover.update_traces(texttemplate="%{value:.2f}", textposition="outside")
    fig_turnover.update_yaxes(autorange="reversed")

    col11, col12, col13 = st.columns(3)

    with col11:
        with st.container(border=True):
            st.plotly_chart(fig_pie)

    with col12:
        with st.container(border=True):
            st.plotly_chart(fig_bar)

    with col13:
        with st.container(border=True):
            st.plotly_chart(fig_bar_brand)

    # brands to target
    low_sales_threshold = brand["TotalSalesDollar"].quantile(0.15)
    high_margin_threshold = brand["ProfitMargin"].quantile(0.85)
    brand["TargetBrand"] = np.where(
        (brand["TotalSalesDollar"] < low_sales_threshold)
        & (brand["ProfitMargin"] > high_margin_threshold),
        "Yes",
        "No",
    )
    brand_performance = brand[
        brand["TotalSalesDollar"] < 5000
    ]  # for better vizualization

    target_brand_chart = px.scatter(
        brand_performance,
        x="TotalSalesDollar",
        y="ProfitMargin",
        hover_data=["Description"],
        color="TargetBrand",
        color_discrete_map={
            "Yes": "#cc5d5d",
            "No": "light blue",
        },
        title="Brands with Low Sales & High Returns",
    )

    left, right = st.columns([1, 2])

    with left:
        with st.container(border=True):
            st.plotly_chart(fig_turnover)

    with right:
        with st.container(border=True):
            st.plotly_chart(target_brand_chart)


build_report()
