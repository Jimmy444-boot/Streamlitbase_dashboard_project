
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import calendar
import plotly.express as px
import io
import altair as alt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Superstore Dashboard")
st.markdown('<style>div.block-container{padding-top:2rem;}<style>', unsafe_allow_html=True)



# --- Initialize empty DataFrames ---
df = pd.DataFrame()
filtered_df = pd.DataFrame()
prev_df = pd.DataFrame()

# --- Upload CSV files ---
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True)

if uploaded_files:  # Only run if at least one file is uploaded
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        buffer = io.BytesIO(bytes_data)
        temp_df = pd.read_csv(buffer, encoding='latin1')
        
        # Normalize column names
        temp_df.columns = temp_df.columns.str.strip().str.lower().str.replace(" ", "_")
        
        # Append to main df
        df = pd.concat([df, temp_df], ignore_index=True)

    # --- Ensure 'order_date' exists ---
    if 'order_date' not in df.columns:
        st.error("CSV must contain an 'order_date' column")
        st.stop()

    # Convert 'order_date' to datetime safely
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')

    # --- FILTERS ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    # 1. Date Filter
    with col1:
        min_date = df['order_date'].min().date()
        max_date = df['order_date'].max().date()
        default_date_range = (min_date, max_date)
        date_range = st.date_input(
            "Date Filter",
            value=default_date_range,
            min_value=min_date,
            max_value=max_date,
            format="MM.DD.YYYY",
        )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    st.markdown("---")
    # 2. Additional filters
    with col2:
        region_option = st.selectbox(
            "Region Filter",
            options=["All"] + sorted(df['region'].dropna().unique().tolist()),
            index=0
        )
    with col3:
        state_option = st.selectbox(
            "state Filter",
            options=["All"] + sorted(df['state'].dropna().unique().tolist()),
            index=0
        )
    with col4:
        category_option = st.selectbox(
            "category Filter",
            options=["All"] + sorted(df['category'].dropna().unique().tolist()),
            index=0
        )

    # --- APPLY FILTERS ---
    filtered_df = df[
        (df['order_date'] >= start_date) &
        (df['order_date'] <= end_date)
    ]
    if region_option != "All":
        filtered_df = filtered_df[filtered_df['region'] == region_option]
    if state_option != "All":
        filtered_df = filtered_df[filtered_df['state'] == state_option]
    if category_option != "All":
        filtered_df = filtered_df[filtered_df['category'] == category_option]

    # --- Previous period ---
    period_length = end_date - start_date
    prev_start_date = start_date - period_length - pd.Timedelta(days=1)
    prev_end_date = start_date - pd.Timedelta(days=1)
    prev_df = df[
        (df['order_date'] >= prev_start_date) &
        (df['order_date'] <= prev_end_date)
    ]

else:
    st.warning("Please upload at least one CSV file to continue.")
    st.stop()

# --- Human readable number format ---
def human_format(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:3.2f}{unit}"
        num /= 1000.0
    return f"{num:.2f}P"

# --- Safe aggregation functions ---
def safe_nunique(df, col):
    return df[col].nunique() if col in df.columns else 0

def safe_sum(df, col):
    return df[col].sum() if col in df.columns else 0

def calc_pct_diff(current, prev):
    return ((current - prev) / prev * 100) if prev != 0 else float('inf')

# --- Calculate metrics ---
current_orders = safe_nunique(filtered_df, "order_id")
prev_orders = safe_nunique(prev_df, "order_id")
orders_diff = calc_pct_diff(current_orders, prev_orders)

current_customers = safe_nunique(filtered_df, "customer_id")
prev_customers = safe_nunique(prev_df, "customer_id")
customers_diff = calc_pct_diff(current_customers, prev_customers)

current_sales = safe_sum(filtered_df, "sales")
prev_sales = safe_sum(prev_df, "sales")
sales_diff = calc_pct_diff(current_sales, prev_sales)

current_quantity = safe_sum(filtered_df, "quantity")
prev_quantity = safe_sum(prev_df, "quantity")
quantity_diff = calc_pct_diff(current_quantity, prev_quantity)

# --- Display metrics ---
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Order count",
    human_format(current_orders),
    delta=f"{orders_diff:+.1f}%",
    help=f"Compared to {prev_start_date.date()} to {prev_end_date.date()}", 
    border=True
)

col2.metric(
    "Customer count",
    human_format(current_customers),
    delta=f"{customers_diff:+.1f}%",
    help=f"Compared to {prev_start_date.date()} to {prev_end_date.date()}", 
    border=True
)

col3.metric(
    "Total sales",
    f"${human_format(current_sales)}",
    delta=f"{sales_diff:+.1f}%",
    help=f"Compared to {prev_start_date.date()} to {prev_end_date.date()}",
    border=True
)

col4.metric(
    "Total quantity",
    human_format(current_quantity),
    delta=f"{quantity_diff:+.1f}%",
    help=f"Compared to {prev_start_date.date()} to {prev_end_date.date()}", 
    border=True
)


#line divider
st.markdown("---")

#creating analysis for each metrics
tabs = st.tabs(["Orders", "Customers", "sales", "quantity", "profit"])


#---------------Tab[0]--------------------------
with tabs[0]:
    filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'])
    filtered_df['ship_date'] = pd.to_datetime(filtered_df['ship_date'])
    filtered_df['Delivery Time (days)'] = (filtered_df['ship_date'] - filtered_df['order_date']).dt.days

    # Average delivery time by ship_mode
    avg_delivery = (
        filtered_df.groupby('ship_mode')['Delivery Time (days)']
        .mean()
        .reset_index()
    )
    avg_delivery['Delivery Time (days)'] = avg_delivery['Delivery Time (days)'].round(0)

    # Order counts by ship_mode
    order_counts = filtered_df['ship_mode'].value_counts().reset_index()
    order_counts.columns = ['ship_mode', 'Order Count']

    # Create columns for side by side layout
    col1, col2 = st.columns(2)
    
    "###### Average Delivery Days by Ship mode"
        
    border_style = """
    border: .2px solid grey;  /* green border */
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    width: 50%;  /* set each box to 30% width */
    margin: 0 10px;  /* horizontal margin for spacing */
    box-sizing: border-box;
    """
    sd_class_avg = filtered_df[filtered_df['ship_mode'] == "Same Day"]['Delivery Time (days)'].mean().round(0)
    first_class_avg = filtered_df[filtered_df['ship_mode'] == "First Class"]['Delivery Time (days)'].mean().round(0)
    second_class_avg = filtered_df[filtered_df['ship_mode'] == "Second Class"]['Delivery Time (days)'].mean().round(0)
    standard_class_avg = filtered_df[filtered_df['ship_mode'] == "Standard Class"]['Delivery Time (days)'].mean().round(0)

    container = f"""
    <div style="
        display: flex;
        justify-content: center;  /* center horizontally */
        margin-bottom: 5px;
        ">
        <div style="{border_style}">
            <h6>Same Day</h6>
            <p style="font-size: 25px; margin: 0;">{int(sd_class_avg)} day(s)</p>
        </div>
        <div style="{border_style}">
            <h6>First Class</h6>
            <p style="font-size: 25px; margin: 0;">{int(first_class_avg)} day(s)</p>
        </div>
        <div style="{border_style}">
            <h6>Second Class</h6>
            <p style="font-size: 25px; margin: 0;">{int(second_class_avg)} day(s)</p>
        </div>
        <div style="{border_style}">
            <h6>Standard Class</h6>
            <p style="font-size: 25px; margin: 0;">{int(standard_class_avg)} day(s)</p>
        </div>
    </div>
    """

    st.markdown(container, unsafe_allow_html=True)
    #line divider
    st.markdown("---")


    col1, col2 = st.columns(2)
    with col1:
        # 1.  Pie chart
        "###### Orders By region"
        labels = filtered_df['sub-category'].value_counts().index
        values = filtered_df['sub-category'].value_counts().values

        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5)])

        fig_pie.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.20,        
                xanchor="center",
                x=0.5,
                font=dict(size=12),  
                itemwidth=30,        
            ),
            margin=dict(t=50, b=100),  
            height=500,                
        )

        
        st.plotly_chart(fig_pie, use_container_width=True)


    with col2:
        #2. Bar chart of orders by day of week
        "###### Avg. Orders by days of week"
        filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'])

        filtered_df['Day'] = filtered_df['order_date'].dt.strftime('%a')

        daily_orders = (
            filtered_df.groupby(['order_date', 'Day'])
            .size()
            .reset_index(name='Orders')
        )

        avg_orders = (
            daily_orders.groupby('Day')['Orders']
            .mean()
            .reindex(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            .reset_index()
        )

        avg_orders.columns = ['Day', 'Average Orders']

        # Plot average orders bar chart
        fig_bar = px.bar(avg_orders, x='Day', y='Average Orders', height=500)
        st.plotly_chart(fig_bar)

    st.markdown("---")


    order_counts = filtered_df.groupby('order_date').agg({'order_id': 'count'}).reset_index()
    order_counts.rename(columns={'order_id': 'Orders'}, inplace=True)

    ship_mode_counts = filtered_df['ship_mode'].value_counts().reset_index()
    ship_mode_counts.columns = ['ship_mode', 'Orders']

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        "###### Orders Over Time"
        line_chart = alt.Chart(order_counts).mark_line(point=False).encode(
            x='order_date:T',
            y='Orders:Q',
            tooltip=['order_date:T', 'Orders:Q']
        ).properties(width=700, height=500)
        st.altair_chart(line_chart, use_container_width=True)

    with col2:
        "###### Preferred ship_mode"
        fig_ship_mode = px.bar(ship_mode_counts, x='ship_mode', y='Orders', height=500)
        fig_ship_mode.update_layout(bargap=0.7)
        st.plotly_chart(fig_ship_mode, use_container_width=True)
    
    st.markdown("---")
    
    #chart 6
    "###### Order Count by Month of the Year"


    # Extract month number
    filtered_df['Month'] = filtered_df['order_date'].dt.month

    # Count daily orders per month
    daily_orders = (
        filtered_df.groupby(['order_date', 'Month'])
        .size()
        .reset_index(name='Orders')
    )

    # Calculate average daily orders per month
    avg_orders = (
        daily_orders.groupby('Month')['Orders']
        .mean()
        .reset_index()
    )

    # Convert month number to month name
    avg_orders['Month'] = avg_orders['Month'].apply(lambda x: calendar.month_name[x])

    fig_sales_month = px.line(
        avg_orders,
        x='Month',
        y='Orders',
        markers=True,
        labels={'Orders': 'Average Orders', 'Month': 'Month'},
        height=500
    )

    # Order x-axis by calendar month
    fig_sales_month.update_xaxes(categoryorder='array', categoryarray=list(calendar.month_name)[1:])

    st.plotly_chart(fig_sales_month, use_container_width=True)


#---------------Tab[1]--------------------------
with tabs[1]:
    #chart 1
    "###### Top 5 Customers by sales"
    # Group by customer_name and category, summing sales
    filtered_df['discount_amt'] = (filtered_df['discount'] * filtered_df['sales']).round(0)
    top_customers_detail = (
        filtered_df.groupby(['customer_name', 'sub-category'])[['sales', 'discount_amt']]
        .sum()
        .reset_index()
    )

    # Get top 5 customers by total sales
    top5_customer_names = (
        top_customers_detail.groupby('customer_name')['sales']
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Filter only those top 5 customers
    top_customers_detail = top_customers_detail[
        top_customers_detail['customer_name'].isin(top5_customer_names)
    ]


    # Create stacked bar chart (each color is a category)
    fig_top5 = px.bar(
        top_customers_detail,
        x='customer_name',
        y='sales',
        color='sub-category',
        labels={'sales': 'Total sales', 'customer_name': 'Customer'},
        height=500, 
        hover_data={
        'sales': ':.0f',           # format quantity without decimals
        'discount_amt': ':.0f',           # format discount with 2 decimals
        'customer_name': False,       # hide repeated x-axis label in tooltip
        'sub-category': True          # show sub-category in tooltip
    }
    )

    # Display chart
    st.plotly_chart(fig_top5, use_container_width=True)

    
    #chart 2
    "###### Top 5 Customers by quantity"
    # Group by customer_nameand category, summing sales
    filtered_df['discount_amt'] = (filtered_df['discount'] * filtered_df['sales']).round(0)
    top_customers_detail = (
        filtered_df.groupby(['customer_name'])[['quantity', 'sales','discount_amt' ]]
        .sum()
        .reset_index()
    )

    # Get top 5 customers by total quantity
    top5_customer_names = (
        top_customers_detail.groupby('customer_name')['quantity']
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Filter only those top 5 customers
    top_customers_detail = top_customers_detail[
        top_customers_detail['customer_name'].isin(top5_customer_names)
    ]

    # Create stacked bar chart (each color is a category)
    fig_top5 = px.bar(
        top_customers_detail,
        x='customer_name',
        y='quantity',
        labels={'quantity': 'Total quantity', 'customer_name': 'Customer'},
        height=500, 
        hover_data={
        'quantity': ':.0f',           # format quantity without decimals
        'sales':':.0f',
        'discount_amt': ':.0f',           # format discount with 2 decimals
        'customer_name': False,       # hide repeated x-axis label in tooltip
    }
    )

    # Display chart
    st.plotly_chart(fig_top5, use_container_width=True)

    #chart 3
    "###### Top 5 Most profitable Customers"
    # Group by customer_nameand category, summing sales
    filtered_df['discount_amt'] = (filtered_df['discount'] * filtered_df['sales']).round(0)
    top_customers_detail = (
        filtered_df.groupby(['customer_name', 'sub-category'])[['profit', 'discount_amt']]
        .sum()
        .reset_index()
    )

    # Get top 5 customers by total sales
    top5_customer_names = (
        top_customers_detail.groupby('customer_name')['profit']
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Filter only those top 5 customers
    top_customers_detail = top_customers_detail[
        top_customers_detail['customer_name'].isin(top5_customer_names)
    ]


    # Create stacked bar chart (each color is a category)
    fig_top5 = px.bar(
        top_customers_detail,
        x='customer_name',
        y='profit',
        color='sub-category',
        labels={'profit': 'Total profit', 'customer_name': 'Customer'},
        height=500, 
        hover_data={
        'profit': ':.0f',           # format quantity without decimals
        'discount_amt': ':.0f',           # format discount with 2 decimals
        'customer_name': False,       # hide repeated x-axis label in tooltip
        'sub-category': True          # show sub-category in tooltip
    }
    )

    # Display chart
    st.plotly_chart(fig_top5, use_container_width=True)

#---------------Tab[2]--------------------------
with tabs[2]:
    #chart 1
    # sales Over Time
    "###### sales Over Time"

    sales_over_time = (
        filtered_df.groupby('order_date')['sales']
        .sum()
        .reset_index()
        .sort_values('order_date')
    )

    fig_sales_time = px.line(
        sales_over_time,
        x='order_date',
        y='sales',
        labels={'sales': 'Total sales', 'order_date': 'Date'},
        height=500, 
    )

    st.plotly_chart(fig_sales_time, use_container_width=True)

    #chart 2
    "###### Most Sold Categories by sales"
    top5_categories = (
    filtered_df.groupby('category')['sales']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    fig_top5_categories = px.bar(
        top5_categories,
        x='category',
        y='sales',
        labels={'quantity': 'Total quantity Sold', 'category': 'category'},
        height=500
    )

    st.plotly_chart(fig_top5_categories, use_container_width=True)


    #chart 3
    "##### Top 5 states by sales"
    top5_states_sales = (
    filtered_df.groupby('state')['sales']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    fig_top5_states_sales = px.bar(
        top5_states_sales,
        x='state',
        y='sales',
        labels={'sales': 'Total sales', 'state': 'state'},
        height=500
    )
    st.plotly_chart(fig_top5_states_sales, use_container_width=True)


    #chart 4
    "###### sales by segment"
    segment_sales = (
    filtered_df.groupby('segment')['sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    )
    fig_segment_sales = px.bar(
        segment_sales,
        x='segment',
        y='sales',
        labels={'sales': 'Total sales', 'segment': 'segment'},
        height=500
    )

    st.plotly_chart(fig_segment_sales, use_container_width=True)


    # Chart 5 
    # sales by ship_mode
    "###### sales by ship_mode"
    profit_by_shipmode = (
        filtered_df.groupby('ship_mode')['sales']
        .sum()
        .reset_index()
        .sort_values(by='sales', ascending=False)
    )

    fig_profit_shipmode = px.bar(
        profit_by_shipmode,
        x='ship_mode',
        y='sales',
        labels={'sales': 'Total sales', 'ship_mode': 'ship_mode'},
        height=500
    )

    st.plotly_chart(fig_profit_shipmode, use_container_width=True)




#---------------Tab[3]--------------------------
with tabs[3]:
    #chart 1
    "###### Total quantity by category"
    top5_categories = (
    filtered_df.groupby('category')['quantity']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    fig_top5_categories = px.bar(
        top5_categories,
        x='category',
        y='quantity',
        labels={'quantity': 'Total quantity', 'category': 'category'},
        height=500
    )

    st.plotly_chart(fig_top5_categories, use_container_width=True)


    #chart 2
    "###### Top 5 states by quantity"
    top5_states_sales = (
    filtered_df.groupby('state')['quantity']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    fig_top5_states_sales = px.bar(
        top5_states_sales,
        x='state',
        y='quantity',
        labels={'quantity': 'Total quantity', 'state': 'state'},
        height=500
    )
    st.plotly_chart(fig_top5_states_sales, use_container_width=True)

    #chart 3
    segment_sales = (
    filtered_df.groupby('segment')['quantity']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    )
    fig_segment_sales = px.bar(
        segment_sales,
        x='segment',
        y='quantity',
        title='sales by segment',
        labels={'quantity': 'Total quantity', 'segment': 'segment'},
        height=500
    )

    st.plotly_chart(fig_segment_sales, use_container_width=True)

    #Chart 4 profit by ship_mode
    "###### quantity by ship_mode"
    profit_by_shipmode = (
        filtered_df.groupby('ship_mode')['quantity']
        .sum()
        .reset_index()
        .sort_values(by='quantity', ascending=False)
    )

    fig_profit_shipmode = px.bar(
        profit_by_shipmode,
        x='ship_mode',
        y='quantity',
        labels={'quantity': 'Total quantity', 'ship_mode': 'ship_mode'},
        height=500
    )

    st.plotly_chart(fig_profit_shipmode, use_container_width=True)



#---------------Tab[3]--------------------------
with tabs[4]: 
    # Chart 1 - category by profit
    top5_categories = (
    filtered_df.groupby('category')['profit']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    # Chart 2 - segment by profit
    segment_sales = (
    filtered_df.groupby('segment')['profit']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    )

    # Create two equal-width columns
    col1, col2 = st.columns(2)

    # First chart - category by profit
    with col1:
        "###### category by profit"
        fig_top5_categories = px.bar(
            top5_categories,
            x='category',
            y='profit',
            text='profit',  # Show values above bars
            labels={'profit': 'Total profit', 'category': 'category'},
            height=500
        )
        fig_top5_categories.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_top5_categories, use_container_width=True)

    # Second chart - segment by profit
    with col2:
        "###### segment by profit"
        fig_segment_sales = px.bar(
            segment_sales,
            x='segment',
            y='profit',
            text='profit',  # Show values above bars
            labels={'profit': 'Total profit', 'segment': 'segment'},
            height=500
        )
        fig_segment_sales.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_segment_sales, use_container_width=True)
    
    # Data for Chart 3 - Top 5 states by profit
    top5_states_profit = (
    filtered_df.groupby('state')['profit']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    # Data for Chart 4 - profit by ship_mode
    profit_by_shipmode = (
        filtered_df.groupby('ship_mode')['profit']
        .sum()
        .reset_index()
        .sort_values(by='profit', ascending=False)
    )

    # Create two equal-width columns
    col1, col2 = st.columns(2)

    # First chart - Bar chart
    with col1:
        "###### Top 5 states by profit"
        fig_top5_states_profit = px.bar(
            top5_states_profit,
            x='state',
            y='profit',
            labels={'profit': 'Total profit', 'state': 'state'},
            height=500
        )
        st.plotly_chart(fig_top5_states_profit, use_container_width=True)

    # Second chart - Pie chart with legend below
    with col2:
        "###### profit by ship_mode"
        fig_profit_shipmode = px.pie(
            profit_by_shipmode,
            names='ship_mode',
            values='profit',
            height=500
        )
        fig_profit_shipmode.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,  # Push legend below chart
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_profit_shipmode, use_container_width=True)
