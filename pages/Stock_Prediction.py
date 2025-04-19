import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pages.utils.model_train import (
    get_data,
    get_rolling_mean,
    get_differencing_order,
    scaling,
    evaluate_model,
    get_forecast,
    inverse_scaling
)
from pages.utils.plotly_figure import plotly_table

# Set up Streamlit page
st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📈 Stock Price Forecasting App")

# Input for stock ticker
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input('Enter Stock Ticker Symbol:', 'AAPL')

st.subheader(f'Predicting Next 30 Days Close Price for: {ticker}')

# Get and process data
close_price = get_data(ticker)

if close_price is None or close_price.empty:
    st.error("Could not retrieve stock data. Please check the ticker symbol.")
else:
    rolling_price = get_rolling_mean(close_price)

    # ✅ Ensure rolling_price has a 'Close' column
    if isinstance(rolling_price, pd.Series):
        rolling_price = rolling_price.to_frame(name='Close')
    elif isinstance(rolling_price, pd.DataFrame) and 'Close' not in rolling_price.columns:
        if rolling_price.shape[1] == 1:
            rolling_price.columns = ['Close']
        else:
            st.error("Rolling data must have a 'Close' column.")
            st.stop()

    # Continue model operations
    differencing_order = get_differencing_order(rolling_price)
    scaled_data, scaler = scaling(rolling_price)
    rmse = evaluate_model(scaled_data, differencing_order)

    st.write("**Model RMSE Score:**", rmse)

    forecast = get_forecast(scaled_data, differencing_order)
    forecast['Close'] = inverse_scaling(scaler, forecast['Close'])

    # Show forecast table
    st.write("#### Forecast Data (Next 30 Days)")
    fig_table = plotly_table(forecast.sort_index().round(3))
    fig_table.update_layout(height=220)
    st.plotly_chart(fig_table, use_container_width=True)

    # Combine actual + forecast for plotting
    combined = pd.concat([rolling_price, forecast])

    # --- Plot forecast with historical ---
    def simple_forecast_plot(actual_data, forecast_data):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=actual_data.index,
            y=actual_data['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))

        fig.add_trace(go.Scatter(
            x=forecast_data.index,
            y=forecast_data['Close'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange')
        ))

        fig.update_layout(
            title='Stock Price Forecast (Next 30 Days)',
            xaxis_title='Date',
            yaxis_title='Close Price',
            height=500
        )

        return fig

    # Show chart
    st.write("#### Forecast Visualization")
    st.plotly_chart(simple_forecast_plot(rolling_price, forecast), use_container_width=True)
