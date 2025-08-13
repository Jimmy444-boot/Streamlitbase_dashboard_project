
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import calendar
from numpy.random import default_rng as rng
import datetime
import plotly.express as px
import io
import altair as alt

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Superstore!!!", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Superstore Dashboard")
st.markdown('<style>div.block-container{padding-top:2rem;}<style>', unsafe_allow_html=True)


#-------Creating a upload button-----------------
uploaded_files = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=True
)

for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("First 5 row of ", uploaded_file.name)
    buffer = io.BytesIO(bytes_data)
    df = pd.read_csv(buffer, encoding='latin1')
    st.write(df.head(5))

# Ensure 'Order Date' is datetime for date filter
df['Order Date'] = pd.to_datetime(df['Order Date'])

# --- FILTERS ---
col1, col2, col3, col4 = st.columns(4)

# 1. Date Filter
with col1:
    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()

# Default date range is full range
    default_date_range = (min_date, max_date)
    date_range = st.date_input(
    "Date Filter",
    value=default_date_range,
    min_value=min_date,
    max_value=max_date,
    format="MM.DD.YYYY",
)

# Check if the date_range input is the full range or user filtered
if date_range == default_date_range or date_range == (max_date, min_date):
    # No filtering, show all data
    filtered_df = df.copy()
else:
    # User applied filter, filter data accordingly
    start_date, end_date = date_range
    filtered_df = df[(df['Order Date'] >= pd.to_datetime(start_date)) & (df['Order Date'] <= pd.to_datetime(end_date))]


# 2. Region Filter
with col2:
    region_option = st.selectbox(
        "Region Filter",
        options=["All"] + sorted(df['Region'].unique().tolist()),
        index=0
    )

# 3. State Filter
with col3:
    state_option = st.selectbox(
        "State Filter",
        options=["All"] + sorted(df['State'].unique().tolist()),
        index=0
    )

# 4. Category Filter
with col4:
    category_option = st.selectbox(
        "Category Filter",
        options=["All"] + sorted(df['Category'].unique().tolist()),
        index=0
    )

# --- APPLY FILTERS ---
start_date, end_date = date_range
filtered_df = df[
    (df['Order Date'] >= pd.Timestamp(start_date)) &
    (df['Order Date'] <= pd.Timestamp(end_date))
]

if region_option != "All":
    filtered_df = filtered_df[filtered_df['Region'] == region_option]

if state_option != "All":
    filtered_df = filtered_df[filtered_df['State'] == state_option]

if category_option != "All":
    filtered_df = filtered_df[filtered_df['Category'] == category_option]




#line divider
st.markdown("---")

# Previous period calculation
period_length = end_date - start_date
prev_start_date = start_date - period_length - datetime.timedelta(days=1)
prev_end_date = start_date - datetime.timedelta(days=1)

prev_df = df[(df['Order Date'] >= pd.Timestamp(prev_start_date)) & (df['Order Date'] <= pd.Timestamp(prev_end_date))]

# Human readable format function
def human_format(num):
    num = float(num)
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    suffixes = ['', 'K', 'M', 'B', 'T']
    return f'{num:.1f}{suffixes[magnitude]}'

# Function to calculate percentage difference safely
def calc_pct_diff(current, previous):
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

# Current and previous metrics
current_orders = filtered_df["Order ID"].nunique()
prev_orders = prev_df["Order ID"].nunique()
orders_diff = calc_pct_diff(current_orders, prev_orders)

current_customers = filtered_df["Customer ID"].nunique()
prev_customers = prev_df["Customer ID"].nunique()
customers_diff = calc_pct_diff(current_customers, prev_customers)

current_sales = filtered_df["Sales"].sum()
prev_sales = prev_df["Sales"].sum()
sales_diff = calc_pct_diff(current_sales, prev_sales)

current_quantity = filtered_df["Quantity"].sum()
prev_quantity = prev_df["Quantity"].sum()
quantity_diff = calc_pct_diff(current_quantity, prev_quantity)

# Display metrics with delta
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Order count",
    human_format(current_orders),
    delta=f"{orders_diff:+.1f}%",
    help=f"Compared to {prev_start_date} to {prev_end_date}", 
    border=True
)

col2.metric(
    "Customer count",
    human_format(current_customers),
    delta=f"{customers_diff:+.1f}%",
    help=f"Compared to {prev_start_date} to {prev_end_date}", 
    border=True
)

col3.metric(
    "Total Sales",
    f"${human_format(current_sales)}",
    delta=f"{sales_diff:+.1f}%",
    help=f"Compared to {prev_start_date} to {prev_end_date}",
    border=True
)

col4.metric(
    "Total Quantity",
    human_format(current_quantity),
    delta=f"{quantity_diff:+.1f}%",
    help=f"Compared to {prev_start_date} to {prev_end_date}", 
    border=True
)

#line divider
st.markdown("---")

#creating analysis for each metrics
tabs = st.tabs(["Orders", "Customers", "Sales", "Quantity", "Profit"])


#---------------Tab[0]--------------------------
with tabs[0]:
    filtered_df['Order Date'] = pd.to_datetime(filtered_df['Order Date'])
    filtered_df['Ship Date'] = pd.to_datetime(filtered_df['Ship Date'])
    filtered_df['Delivery Time (days)'] = (filtered_df['Ship Date'] - filtered_df['Order Date']).dt.days

    # Average delivery time by Ship Mode
    avg_delivery = (
        filtered_df.groupby('Ship Mode')['Delivery Time (days)']
        .mean()
        .reset_index()
    )
    avg_delivery['Delivery Time (days)'] = avg_delivery['Delivery Time (days)'].round(0)

    # Order counts by Ship Mode
    order_counts = filtered_df['Ship Mode'].value_counts().reset_index()
    order_counts.columns = ['Ship Mode', 'Order Count']

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
    sd_class_avg = filtered_df[filtered_df['Ship Mode'] == "Same Day"]['Delivery Time (days)'].mean().round(0)
    first_class_avg = filtered_df[filtered_df['Ship Mode'] == "First Class"]['Delivery Time (days)'].mean().round(0)
    second_class_avg = filtered_df[filtered_df['Ship Mode'] == "Second Class"]['Delivery Time (days)'].mean().round(0)
    standard_class_avg = filtered_df[filtered_df['Ship Mode'] == "Standard Class"]['Delivery Time (days)'].mean().round(0)

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
        labels = filtered_df['Sub-Category'].value_counts().index
        values = filtered_df['Sub-Category'].value_counts().values

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
        filtered_df['Order Date'] = pd.to_datetime(filtered_df['Order Date'])

        filtered_df['Day'] = filtered_df['Order Date'].dt.strftime('%a')

        daily_orders = (
            filtered_df.groupby(['Order Date', 'Day'])
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


    order_counts = filtered_df.groupby('Order Date').agg({'Order ID': 'count'}).reset_index()
    order_counts.rename(columns={'Order ID': 'Orders'}, inplace=True)

    ship_mode_counts = filtered_df['Ship Mode'].value_counts().reset_index()
    ship_mode_counts.columns = ['Ship Mode', 'Orders']

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        "###### Orders Over Time"
        line_chart = alt.Chart(order_counts).mark_line(point=False).encode(
            x='Order Date:T',
            y='Orders:Q',
            tooltip=['Order Date:T', 'Orders:Q']
        ).properties(width=700, height=500)
        st.altair_chart(line_chart, use_container_width=True)

    with col2:
        "###### Preferred Ship Mode"
        fig_ship_mode = px.bar(ship_mode_counts, x='Ship Mode', y='Orders', height=500)
        fig_ship_mode.update_layout(bargap=0.7)
        st.plotly_chart(fig_ship_mode, use_container_width=True)
    
    st.markdown("---")
    
    #chart 6
    "###### Order Count by Month of the Year"


    # Extract month number
    filtered_df['Month'] = filtered_df['Order Date'].dt.month

    # Count daily orders per month
    daily_orders = (
        filtered_df.groupby(['Order Date', 'Month'])
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
    "###### Top 5 Customers by Sales"
    # Group by Customer Name and Category, summing Sales
    filtered_df['Discount_amt'] = (filtered_df['Discount'] * filtered_df['Sales']).round(0)
    top_customers_detail = (
        filtered_df.groupby(['Customer Name', 'Sub-Category'])[['Sales', 'Discount_amt']]
        .sum()
        .reset_index()
    )

    # Get top 5 customers by total sales
    top5_customer_names = (
        top_customers_detail.groupby('Customer Name')['Sales']
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Filter only those top 5 customers
    top_customers_detail = top_customers_detail[
        top_customers_detail['Customer Name'].isin(top5_customer_names)
    ]


    # Create stacked bar chart (each color is a category)
    fig_top5 = px.bar(
        top_customers_detail,
        x='Customer Name',
        y='Sales',
        color='Sub-Category',
        labels={'Sales': 'Total Sales', 'Customer Name': 'Customer'},
        height=500, 
        hover_data={
        'Sales': ':.0f',           # format quantity without decimals
        'Discount_amt': ':.0f',           # format discount with 2 decimals
        'Customer Name': False,       # hide repeated x-axis label in tooltip
        'Sub-Category': True          # show sub-category in tooltip
    }
    )

    # Display chart
    st.plotly_chart(fig_top5, use_container_width=True)

    
    #chart 2
    "###### Top 5 Customers by Quantity"
    # Group by Customer Name and Category, summing Sales
    filtered_df['Discount_amt'] = (filtered_df['Discount'] * filtered_df['Sales']).round(0)
    top_customers_detail = (
        filtered_df.groupby(['Customer Name'])[['Quantity', 'Sales','Discount_amt' ]]
        .sum()
        .reset_index()
    )

    # Get top 5 customers by total Quantity
    top5_customer_names = (
        top_customers_detail.groupby('Customer Name')['Quantity']
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Filter only those top 5 customers
    top_customers_detail = top_customers_detail[
        top_customers_detail['Customer Name'].isin(top5_customer_names)
    ]

    # Create stacked bar chart (each color is a category)
    fig_top5 = px.bar(
        top_customers_detail,
        x='Customer Name',
        y='Quantity',
        labels={'Quantity': 'Total Quantity', 'Customer Name': 'Customer'},
        height=500, 
        hover_data={
        'Quantity': ':.0f',           # format quantity without decimals
        'Sales':':.0f',
        'Discount_amt': ':.0f',           # format discount with 2 decimals
        'Customer Name': False,       # hide repeated x-axis label in tooltip
    }
    )

    # Display chart
    st.plotly_chart(fig_top5, use_container_width=True)

    #chart 3
    "###### Top 5 Most Profitable Customers"
    # Group by Customer Name and Category, summing Sales
    filtered_df['Discount_amt'] = (filtered_df['Discount'] * filtered_df['Sales']).round(0)
    top_customers_detail = (
        filtered_df.groupby(['Customer Name', 'Sub-Category'])[['Profit', 'Discount_amt']]
        .sum()
        .reset_index()
    )

    # Get top 5 customers by total sales
    top5_customer_names = (
        top_customers_detail.groupby('Customer Name')['Profit']
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )

    # Filter only those top 5 customers
    top_customers_detail = top_customers_detail[
        top_customers_detail['Customer Name'].isin(top5_customer_names)
    ]


    # Create stacked bar chart (each color is a category)
    fig_top5 = px.bar(
        top_customers_detail,
        x='Customer Name',
        y='Profit',
        color='Sub-Category',
        labels={'Profit': 'Total Profit', 'Customer Name': 'Customer'},
        height=500, 
        hover_data={
        'Profit': ':.0f',           # format quantity without decimals
        'Discount_amt': ':.0f',           # format discount with 2 decimals
        'Customer Name': False,       # hide repeated x-axis label in tooltip
        'Sub-Category': True          # show sub-category in tooltip
    }
    )

    # Display chart
    st.plotly_chart(fig_top5, use_container_width=True)

#---------------Tab[2]--------------------------
with tabs[2]:
    #chart 1
    # Sales Over Time
    "###### Sales Over Time"

    sales_over_time = (
        filtered_df.groupby('Order Date')['Sales']
        .sum()
        .reset_index()
        .sort_values('Order Date')
    )

    fig_sales_time = px.line(
        sales_over_time,
        x='Order Date',
        y='Sales',
        labels={'Sales': 'Total Sales', 'Order Date': 'Date'},
        height=500, 
    )

    st.plotly_chart(fig_sales_time, use_container_width=True)

    #chart 2
    "###### Most Sold Categories by Sales"
    top5_categories = (
    filtered_df.groupby('Category')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    fig_top5_categories = px.bar(
        top5_categories,
        x='Category',
        y='Sales',
        labels={'Quantity': 'Total Quantity Sold', 'Category': 'Category'},
        height=500
    )

    st.plotly_chart(fig_top5_categories, use_container_width=True)


    #chart 3
    "##### Top 5 States by Sales"
    top5_states_sales = (
    filtered_df.groupby('State')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    fig_top5_states_sales = px.bar(
        top5_states_sales,
        x='State',
        y='Sales',
        labels={'Sales': 'Total Sales', 'State': 'State'},
        height=500
    )
    st.plotly_chart(fig_top5_states_sales, use_container_width=True)


    #chart 4
    "###### Sales by Segment"
    segment_sales = (
    filtered_df.groupby('Segment')['Sales']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    )
    fig_segment_sales = px.bar(
        segment_sales,
        x='Segment',
        y='Sales',
        labels={'Sales': 'Total Sales', 'Segment': 'Segment'},
        height=500
    )

    st.plotly_chart(fig_segment_sales, use_container_width=True)


    # Chart 5 
    # Sales by Ship Mode
    "###### Sales by Ship Mode"
    profit_by_shipmode = (
        filtered_df.groupby('Ship Mode')['Sales']
        .sum()
        .reset_index()
        .sort_values(by='Sales', ascending=False)
    )

    fig_profit_shipmode = px.bar(
        profit_by_shipmode,
        x='Ship Mode',
        y='Sales',
        labels={'Sales': 'Total Sales', 'Ship Mode': 'Ship Mode'},
        height=500
    )

    st.plotly_chart(fig_profit_shipmode, use_container_width=True)




#---------------Tab[3]--------------------------
with tabs[3]:
    #chart 1
    "###### Total Quantity by Category"
    top5_categories = (
    filtered_df.groupby('Category')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    fig_top5_categories = px.bar(
        top5_categories,
        x='Category',
        y='Quantity',
        labels={'Quantity': 'Total Quantity', 'Category': 'Category'},
        height=500
    )

    st.plotly_chart(fig_top5_categories, use_container_width=True)


    #chart 2
    "###### Top 5 States by Quantity"
    top5_states_sales = (
    filtered_df.groupby('State')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    fig_top5_states_sales = px.bar(
        top5_states_sales,
        x='State',
        y='Quantity',
        labels={'Quantity': 'Total Quantity', 'State': 'State'},
        height=500
    )
    st.plotly_chart(fig_top5_states_sales, use_container_width=True)

    #chart 3
    segment_sales = (
    filtered_df.groupby('Segment')['Quantity']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    )
    fig_segment_sales = px.bar(
        segment_sales,
        x='Segment',
        y='Quantity',
        title='Sales by Segment',
        labels={'Quantity': 'Total Quantity', 'Segment': 'Segment'},
        height=500
    )

    st.plotly_chart(fig_segment_sales, use_container_width=True)

    #Chart 4 Profit by Ship Mode
    "###### Quantity by Ship Mode"
    profit_by_shipmode = (
        filtered_df.groupby('Ship Mode')['Quantity']
        .sum()
        .reset_index()
        .sort_values(by='Quantity', ascending=False)
    )

    fig_profit_shipmode = px.bar(
        profit_by_shipmode,
        x='Ship Mode',
        y='Quantity',
        labels={'Quantity': 'Total Quantity', 'Ship Mode': 'Ship Mode'},
        height=500
    )

    st.plotly_chart(fig_profit_shipmode, use_container_width=True)



#---------------Tab[3]--------------------------
with tabs[4]: 
    # Chart 1 - Category by Profit
    top5_categories = (
    filtered_df.groupby('Category')['Profit']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    # Chart 2 - Segment by Profit
    segment_sales = (
    filtered_df.groupby('Segment')['Profit']
    .sum()
    .sort_values(ascending=False)
    .reset_index()
    )

    # Create two equal-width columns
    col1, col2 = st.columns(2)

    # First chart - Category by Profit
    with col1:
        "###### Category by Profit"
        fig_top5_categories = px.bar(
            top5_categories,
            x='Category',
            y='Profit',
            text='Profit',  # Show values above bars
            labels={'Profit': 'Total Profit', 'Category': 'Category'},
            height=500
        )
        fig_top5_categories.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_top5_categories, use_container_width=True)

    # Second chart - Segment by Profit
    with col2:
        "###### Segment by Profit"
        fig_segment_sales = px.bar(
            segment_sales,
            x='Segment',
            y='Profit',
            text='Profit',  # Show values above bars
            labels={'Profit': 'Total Profit', 'Segment': 'Segment'},
            height=500
        )
        fig_segment_sales.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_segment_sales, use_container_width=True)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Data for Chart 3 - Top 5 States by Profit
    top5_states_profit = (
    filtered_df.groupby('State')['Profit']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    )

    # Data for Chart 4 - Profit by Ship Mode
    profit_by_shipmode = (
        filtered_df.groupby('Ship Mode')['Profit']
        .sum()
        .reset_index()
        .sort_values(by='Profit', ascending=False)
    )

    # Create two equal-width columns
    col1, col2 = st.columns(2)

    # First chart - Bar chart
    with col1:
        "###### Top 5 States by Profit"
        fig_top5_states_profit = px.bar(
            top5_states_profit,
            x='State',
            y='Profit',
            labels={'Profit': 'Total Profit', 'State': 'State'},
            height=500
        )
        st.plotly_chart(fig_top5_states_profit, use_container_width=True)

    # Second chart - Pie chart with legend below
    with col2:
        "###### Profit by Ship Mode"
        fig_profit_shipmode = px.pie(
            profit_by_shipmode,
            names='Ship Mode',
            values='Profit',
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
