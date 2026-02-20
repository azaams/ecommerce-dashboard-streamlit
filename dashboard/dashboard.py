import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

# Set global Seaborn theme
sns.set_theme(style='dark')

def create_sum_order_items_df(df):
    """
    Creates a summary of ordered items grouped by product category.
    """
    sum_orders_items_df = df.groupby("product_category").order_id.count().sort_values(ascending=False).reset_index()
    return sum_orders_items_df

def create_monthly_orders_df(df):
    """
    Creates a summary of monthly orders and total revenue.
    """
    monthly_orders_df = df.resample(rule='ME', on='order_purchase_timestamp').agg({
        "order_id": "nunique",
        "price": "sum"
    })
    monthly_orders_df.index = monthly_orders_df.index.strftime('%Y-%m')
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)
    return monthly_orders_df

def create_bystate_df(df):
    """
    Creates a summary of unique customers by state.
    """
    bystate_df = df.groupby(by="customer_state").customer_id.nunique().reset_index()
    bystate_df = bystate_df.rename(columns={
        "customer_id": "customer_count"
    })
    return bystate_df

def create_rfm_df(df):
    """
    Creates RFM (Recency, Frequency, Monetary) analysis dataframe.
    """
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max",
        "order_id": "nunique",
        "price": "sum"
    })
    rfm_df.columns = ["customer_id", "max_order_timestamp", "frequency", "monetary"]
    rfm_df["max_order_timestamp"] = rfm_df["max_order_timestamp"].dt.date
    recent_date = df["order_purchase_timestamp"].dt.date.max()
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    return rfm_df

def perform_manual_clustering(rfm_df):
    """
    Perform manual grouping based on RFM scores.
    """
    # Define scoring for R, F, M
    rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
    rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
    rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)

    # Normalize scores
    rfm_df['r_score'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 5
    rfm_df['f_score'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 5
    rfm_df['m_score'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 5

    rfm_df['rfm_score'] = 0.4 * rfm_df['r_score'] + 0.3 * rfm_df['f_score'] + 0.3 * rfm_df['m_score']
    
    # Define Segments
    def segment_customer(score):
        if score >= 4:
            return "Champion"
        elif score >= 3:
            return "Loyal"
        elif score >= 2:
            return "At Risk"
        else:
            return "Lost"

    rfm_df['customer_segment'] = rfm_df['rfm_score'].apply(segment_customer)
    return rfm_df

def perform_spending_binning(df):
    """
    Perform binning on monetary values to categorize spending levels.
    """
    bins = [0, 50, 200, 1000, df['price'].max()]
    labels = ['Budget', 'Standard', 'Premium', 'Luxury']
    df['spending_category'] = pd.cut(df['price'], bins=bins, labels=labels, include_lowest=True)
    return df

def load_and_preprocess_data(file_path):
    """
    Loads data and performs initial preprocessing.
    """
    df = pd.read_csv(file_path)
    
    # Convert date columns to datetime objects
    datetime_columns = [
        "order_purchase_timestamp", 
        "order_approved_at", 
        "order_delivered_carrier_date", 
        "order_delivered_customer_date", 
        "order_estimated_delivery_date"
    ]
    for column in datetime_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column])
            
    # Sort data by purchase timestamp
    df.sort_values(by="order_purchase_timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def main():
    # Load main dataset
    all_df = load_and_preprocess_data("main_data.csv")
    
    # Date filter boundaries
    min_date = all_df["order_purchase_timestamp"].min().date()
    max_date = all_df["order_purchase_timestamp"].max().date()
    
    # Sidebar: Filtering
    with st.sidebar:
        start_date, end_date = st.date_input(
            label='Time Range',
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date]
        )
    
    # Apply time range filter
    main_df = all_df[
        (all_df["order_purchase_timestamp"].dt.date >= start_date) & 
        (all_df["order_purchase_timestamp"].dt.date <= end_date)
    ]
    
    # Prepare component dataframes
    sum_order_items_df = create_sum_order_items_df(main_df)
    monthly_orders_df = create_monthly_orders_df(main_df)
    bystate_df = create_bystate_df(main_df)
    rfm_df = create_rfm_df(main_df)
    
    # Advanced Analysis: Clustering
    rfm_df = perform_manual_clustering(rfm_df)
    main_df = perform_spending_binning(main_df)
    
    # Dashboard Header
    st.header('E-Commerce Public Dashboard')
    
    # Section 1: Monthly Orders & Revenue
    st.subheader('Monthly Performance')
    
    col1, col2 = st.columns(2)
    with col1:
        total_orders = monthly_orders_df.order_count.sum()
        st.metric("Total Orders", value=total_orders)
    with col2:
        total_revenue = format_currency(monthly_orders_df.revenue.sum(), "USD", locale='en_US')
        st.metric("Total Revenue", value=total_revenue)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        monthly_orders_df["order_purchase_timestamp"],
        monthly_orders_df["revenue"],
        marker='o', 
        linewidth=2,
        color="#90CAF9"
    )
    ax.set_title("Monthly Revenue Trend", fontsize=20)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    st.pyplot(fig)
    
    # Section 2: Product Performance
    st.subheader("Product Performance")
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 10))
    colors = ["#90CAF9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]
    
    # Best Performing Products
    sns.barplot(
        x="order_id", 
        y="product_category", 
        data=sum_order_items_df.head(5), 
        palette=colors, 
        ax=ax[0],
        hue="product_category",
        legend=False
    )
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Number of Sales", fontsize=15)
    ax[0].set_title("Top 5 Best Performing Products", loc="center", fontsize=20)
    ax[0].tick_params(axis='y', labelsize=12)
    ax[0].tick_params(axis='x', labelsize=12)
    
    # Worst Performing Products
    sns.barplot(
        x="order_id", 
        y="product_category", 
        data=sum_order_items_df.sort_values(by="order_id", ascending=True).head(5), 
        palette=colors, 
        ax=ax[1],
        hue="product_category",
        legend=False
    )
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Number of Sales", fontsize=15)
    ax[1].invert_xaxis()
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].set_title("Top 5 Worst Performing Products", loc="center", fontsize=20)
    ax[1].tick_params(axis='y', labelsize=12)
    ax[1].tick_params(axis='x', labelsize=12)
    
    st.pyplot(fig)
    
    # Section 3: Geospatial Analysis
    st.subheader("Geospatial Analysis")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    state_rank_df = bystate_df.sort_values(by="customer_count", ascending=False).head(10)
    sns.barplot(
        x="customer_count", 
        y="customer_state",
        data=state_rank_df,
        palette="Blues_d",
        ax=ax,
        hue="customer_state",
        legend=False
    )
    ax.set_title("Customer Density by Top 10 States", loc="center", fontsize=20)
    ax.set_ylabel(None)
    ax.set_xlabel("Total Customers", fontsize=15)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    st.pyplot(fig)
    
    # Section 4: Customer Segmentation (Clustering)
    st.subheader("Customer Segmentation & Binning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Manual Segmentation (RFM Based)")
        fig, ax = plt.subplots(figsize=(10, 6))
        segment_df = rfm_df['customer_segment'].value_counts().reset_index()
        segment_df.columns = ['Segment', 'Count']
        sns.barplot(
            x='Count', 
            y='Segment', 
            data=segment_df, 
            palette="viridis", 
            ax=ax,
            hue='Segment',
            legend=False
        )
        ax.set_title("Customer Segments", fontsize=15)
        st.pyplot(fig)

    with col2:
        st.write("#### Spending Categories (Binning)")
        fig, ax = plt.subplots(figsize=(10, 6))
        spending_df = main_df['spending_category'].value_counts().reset_index()
        spending_df.columns = ['Category', 'Count']
        sns.barplot(
            x='Count', 
            y='Category', 
            data=spending_df, 
            palette="magma", 
            ax=ax,
            hue='Category',
            legend=False
        )
        ax.set_title("Spending Categories", fontsize=15)
        st.pyplot(fig)

    # Section 5: RFM Parameters Overview
    st.subheader("RFM Parameters Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_recency = round(rfm_df.recency.mean(), 1)
        st.metric("Avg Recency (days)", value=avg_recency)
    with col2:
        avg_frequency = round(rfm_df.frequency.mean(), 2)
        st.metric("Avg Frequency", value=avg_frequency)
    with col3:
        avg_monetary = format_currency(rfm_df.monetary.mean(), "USD", locale='en_US')
        st.metric("Avg Monetary Value", value=avg_monetary)
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    bar_color = "#90CAF9"
    
    # Recency
    sns.barplot(y="recency", x="customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5), color=bar_color, ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Customer ID", fontsize=15)
    ax[0].set_title("Top 5 by Recency (days)", loc="center", fontsize=20)
    ax[0].tick_params(axis='x', labelsize=10, rotation=45)
    ax[0].tick_params(axis='y', labelsize=12)
    
    # Frequency
    sns.barplot(y="frequency", x="customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5), color=bar_color, ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Customer ID", fontsize=15)
    ax[1].set_title("Top 5 by Frequency", loc="center", fontsize=20)
    ax[1].tick_params(axis='x', labelsize=10, rotation=45)
    ax[1].tick_params(axis='y', labelsize=12)
    
    # Monetary
    sns.barplot(y="monetary", x="customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5), color=bar_color, ax=ax[2])
    ax[2].set_ylabel(None)
    ax[2].set_xlabel("Customer ID", fontsize=15)
    ax[2].set_title("Top 5 by Monetary", loc="center", fontsize=20)
    ax[2].tick_params(axis='x', labelsize=10, rotation=45)
    ax[2].tick_params(axis='y', labelsize=12)
    
    st.pyplot(fig)
    
    st.caption('Copyright (C) 2026')

if __name__ == "__main__":
    main()
