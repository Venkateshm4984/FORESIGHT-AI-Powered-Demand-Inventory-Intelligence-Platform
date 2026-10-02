# FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

**Domain:** Data Science & Analytics  
**Project type:** Predictive analytics and decision-support dashboard  
**Author:** Venkatesh M  
**Date:** September 2026

## 1. Abstract
FORESIGHT is an analytics prototype that helps businesses plan stock using historical SKU-level sales. It combines demand forecasting, inventory risk identification, and interpretable replenishment calculations in an interactive dashboard. The app accepts a sample dataset or user-uploaded CSV, generates forecasts, and presents stockout/reorder alerts and suggested replenishment quantities.

## 2. Problem Statement
Retailers can experience excess stock, tying up working capital, or insufficient stock, causing lost sales and poor customer experience. Manual spreadsheet-based planning may not scale across products and changing demand. FORESIGHT demonstrates how historical sales data and machine learning can support more consistent demand and inventory decisions.

## 3. Objectives
- Explore and visualize historical sales patterns.
- Forecast near-term demand for each SKU.
- Identify products at stockout risk or nearing a reorder threshold.
- Estimate suggested order quantities and indicative inventory value.
- Provide downloadable outputs for further analysis.

## 4. Technology Stack
- Python — application and analytics logic
- Pandas / NumPy — data preparation and numerical calculations
- Scikit-learn — Random Forest regression
- Plotly — interactive visualizations
- Streamlit — web dashboard
- CSV — lightweight data input and output

## 5. System Architecture
1. **Data input:** bundled demonstration CSV or uploaded CSV.
2. **Preprocessing:** normalize column names, parse dates, convert numeric fields, remove invalid rows, and sort by SKU/date.
3. **Feature engineering:** lagged sales (1, 7, 14 days), rolling means (7, 14 days), weekday, day of month, and month.
4. **Forecasting:** one Random Forest model per SKU with sufficient history; moving-average baseline for shorter histories.
5. **Inventory intelligence:** compare current stock with lead-time demand and safety-stock-adjusted reorder point.
6. **Presentation:** dashboard metrics, charts, tables, and CSV downloads.

## 6. Data Description
The included sample dataset is synthetic and contains 120 days of daily sales for eight product SKUs. Fields include date, SKU, category, units sold, stock quantity, supplier lead time, and unit cost. Replace the sample with authorized real data before drawing business conclusions.

## 7. Methodology
For SKU \(i\), the model uses historical lags and calendar features to estimate future daily demand. Forecasts are recursively generated across the selected horizon. The application calculates:
- **Average daily forecast:** total forecast units / forecast horizon.
- **Reorder point:** average daily forecast × (lead time + safety coverage).
- **Suggested order quantity:** max(0, ceiling(reorder point − current stock)).
- **Indicative inventory value:** current stock × unit cost.

Risk labels are:
- **Stockout risk:** current stock is below estimated demand over supplier lead time.
- **Reorder soon:** stock is below the reorder point but not below lead-time demand.
- **Healthy:** stock meets or exceeds the reorder point.

## 8. Dashboard Modules
- **Overview:** KPIs, historical sales trend, inventory health distribution, priority SKU watchlist.
- **Demand Forecast:** actual sales versus forecast for a selected SKU.
- **Inventory Recommendations:** risk-filtered SKU table, suggested order quantity/value, downloadable CSV.
- **Data Explorer:** cleaned records and data quality summary.

## 9. How to Execute
Install Python 3.10+, install packages with `pip install -r requirements.txt`, then run `streamlit run app.py`. Open the local URL shown in the terminal, normally `http://localhost:8501`. Full Windows and macOS/Linux instructions are in `README.md`.

## 10. Testing and Evaluation Plan
Before operational use, split each SKU's data chronologically into training and test periods and compare the forecast with a naïve baseline using MAE and WAPE. Also test missing values, duplicate dates, short history, zero demand, new SKUs, and unusual promotions. The current prototype is designed for demonstration and does not claim production-grade accuracy.

## 11. Limitations and Future Scope
- Synthetic data is illustrative, not evidence of real retail performance.
- Forecasting currently uses a simple feature-based model and recursive prediction.
- No automatic purchase-order creation, ERP integration, or user authentication.
- Add backtesting/model comparison, confidence intervals, holiday and promotion features, stockout-aware demand estimation, supplier constraints, service-level targets, and scheduled data refreshes.

## 12. Conclusion
FORESIGHT demonstrates an end-to-end data science workflow: data cleaning, exploratory visualization, machine-learning-based demand forecasting, inventory risk classification, and actionable recommendations. It provides a base for a more rigorous version using real sales and inventory data, formal model evaluation, and business-specific replenishment constraints.
