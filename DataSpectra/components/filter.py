import streamlit as st
import numpy as np

def render_global_filter_widgets(data):
    st.sidebar.title("Global Data Filters")
    filter_container = st.sidebar.container()
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_columns = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    all_filterable_columns = numeric_columns + categorical_columns + datetime_columns
    if not all_filterable_columns:
        return data
    filter_cols = filter_container.multiselect(
        "Select global columns to filter by:",
        options=all_filterable_columns,
        key="global_filter_cols"
    )
    filtered_data = data.copy()
    for col in filter_cols:
        if col in numeric_columns:
            min_val, max_val = float(filtered_data[col].min()), float(filtered_data[col].max())
            if min_val < max_val:
                slider_range = filter_container.slider(
                    f"Select range for '{col}'",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"global_slider_{col}"
                )
                filtered_data = filtered_data[
                    (filtered_data[col] >= slider_range[0]) & (filtered_data[col] <= slider_range[1])
                ]
            else:
                filter_container.info(f"Column '{col}' has only one unique value ({min_val}) and cannot be filtered with a range slider.")
        elif col in categorical_columns:
            unique_vals = sorted(filtered_data[col].astype(str).unique())
            selected_vals = filter_container.multiselect(
                f"Select values for '{col}'",
                options=unique_vals,
                key=f"global_multi_{col}"
            )
            if selected_vals:
                filtered_data = filtered_data[filtered_data[col].isin(selected_vals)]
        elif col in datetime_columns:
            min_date = filtered_data[col].min()
            max_date = filtered_data[col].max()
            if min_date.date() < max_date.date():
                date_range = filter_container.date_input(
                    f"Select date range for '{col}'",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key=f"global_date_{col}"
                )
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_data = filtered_data[
                        (filtered_data[col].dt.date >= start_date) & (filtered_data[col].dt.date <= end_date)]
            else:
                filter_container.info(f"Column '{col}' contains dates from a single day and cannot be filtered with a date range.")
    return filtered_data