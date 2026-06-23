# Vendor Sales & Inventory Analytics Dashboard

**Stack:** SQL (star schema, RFM-style aggregation) → Python (pandas, feature engineering) → Streamlit + Plotly (interactive dashboard)

**🔗 Live demo:** [vendorsalessummary-gow4qlomjfso89elxtekdg.streamlit.app](https://vendorsalessummary-gow4qlomjfso89elxtekdg.streamlit.app/)

## Overview

This project analyzes vendor purchasing, sales, and inventory data for a retail/distribution business, surfacing which vendors and brands drive profitability, where inventory is moving too slowly, and which low-visibility products are quietly outperforming on margin.

The pipeline pulls raw transactional data from a SQLite database, joins and aggregates it into vendor/brand-level metrics via SQL, enriches it in Python with derived KPIs (gross profit, profit margin, stock turnover, sales-to-purchase ratio), and renders the result as an interactive Streamlit dashboard.

## Dataset

- **8,561** vendor-brand line items analyzed (after filtering to profitable, sold-through SKUs)
- **119** distinct vendors
- **7,706** distinct brands

## Key Metrics

| Metric | Value |
|---|---|
| Total Sales | $441.14M |
| Total Purchases | $307.21M |
| Gross Profit | $133.92M |
| Average Profit Margin | 38.72% |
| Unsold Inventory Value | $2.80M |

## Findings

### 1. Vendor spend is highly concentrated
The top 10 vendors account for **65.7%** of all purchase dollars — a classic Pareto distribution. **Diageo North America** alone drives 16.3% of purchases ($50.1M) and is also the top revenue generator ($68.0M), followed by Martignetti Companies and Pernod Ricard USA. This concentration is useful leverage in vendor negotiations but also represents supply-chain risk if any top vendor is disrupted. <br>
<div align="center">
  <img src="https://github.com/mahesh-bdmi/vendor_sales_summary/blob/master/images/vendorshare.png" alt="Vendor purchases share" />
</div>

### 2. A handful of flagship brands carry the top vendors
Jack Daniels No. 7, Tito's Handmade Vodka, and Grey Goose Vodka are the three highest-revenue SKUs (each $7.2M–$8.0M), mirroring the vendor rankings above — Diageo, Martignetti, and Pernod Ricard's positions are largely built on a small number of marquee spirits brands.<br>
<div align="center">
  <img src="https://github.com/mahesh-bdmi/vendor_sales_summary/blob/master/images/topbrands.png" alt="Top Brands" />
</div>

### 3. A long tail of vendors shows weak inventory turnover
The 10 vendors with the lowest stock turnover (0.73–0.88, meaning under 90% of purchased units have sold through) are almost all small, niche suppliers — Park Street Imports, Dunn Wine Brokers, Tamworth Distilling — with modest purchase volumes ($900–$140K each). Individually low-risk, but collectively they represent tied-up capital and shelf space that could be reallocated.
<br>
<div align="center">
  <img src="https://github.com/mahesh-bdmi/vendor_sales_summary/blob/master/images/lowturnover.png" alt="Vendors with low inventory turnover" />
</div>

### 4. 198 brands are high-margin, low-visibility "hidden gems"
These brands have low total sales (under ~$560) but profit margins above 65%, many clustering near 95–99% (e.g., The Club Strawberry Margarita, Bacardi Oakheart Spiced, Chi Chi's Chocolate Malt RTD). The margin is already excellent — the bottleneck is volume and visibility, making these strong candidates for targeted marketing or improved shelf placement.
<br>
<div align="center">
  <img src="https://github.com/mahesh-bdmi/vendor_sales_summary/blob/master/images/gems.png" alt="Brands needing promotion/to be present in more numbers on the shelf" />
</div>

## Skills Demonstrated

- **SQL:** CTEs, multi-table joins, aggregation, star-schema-style modeling of sales/purchase facts against vendor/brand dimensions
- **Python:** pandas feature engineering, vectorized divide-by-zero-safe ratio calculations, data cleaning
- **Data Visualization:** Plotly Express (donut, gradient bar, scatter with reference lines), theme-aware styling (light/dark mode support)
- **Dashboarding:** Streamlit, caching strategy (`st.cache_data`), defensive pipeline design (auto-rebuilds derived data if missing)

## Possible Next Steps

- Build a vendor risk score combining purchase concentration + turnover into a single ranked list
- Time-series view of sales/margin trends per vendor (if historical data is available)
- A/B test shelf placement changes for the 198 flagged "hidden gem" brands and measure lift
