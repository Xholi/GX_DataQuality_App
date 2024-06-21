import streamlit as st
import pandas as pd
import re
import numbers
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import tempfile as tmp
import pymssql

# Function definitions
def validate_column_type(df, column_name, expected_type):
    if column_name in df.columns:
        return df[column_name].map(lambda x: isinstance(x, expected_type))
    else:
        st.warning(f"Column {column_name} is missing from the dataframe.")
        return pd.Series([False] * len(df))

def validate_column_regex(df, column_name, regex):
    if column_name in df.columns:
        pattern = re.compile(regex)
        return df[column_name].map(lambda x: bool(pattern.match(str(x))))
    else:
        st.warning(f"Column {column_name} is missing from the dataframe.")
        return pd.Series([False] * len(df))

def validate_column_values_in_set(df, column_name, valid_set):
    if column_name in df.columns:
        return df[column_name].isin(valid_set)
    else:
        st.warning(f"Column {column_name} is missing from the dataframe.")
        return pd.Series([False] * len(df))

def calculate_total_validation_percentage(validation_results):
    total_checks = len(validation_results)
    total_passed = sum(result.all() for result in validation_results.values())
    return (total_passed / total_checks) * 100

def calculate_duplication_percentage(df, column_name):
    if column_name in df.columns:
        duplicate_count = df.duplicated(subset=[column_name]).sum()
        total_count = len(df)
        return (duplicate_count / total_count) * 100
    else:
        st.warning(f"Column {column_name} is missing from the dataframe.")
        return 0

def calculate_completeness(data):
    complete_entries = 0
    for index, row in data.iterrows():
        property_value = row['PROPERTY_VALUE']
        property_uom = row['PROPERTY_UOM']
        data_type_rules = row['DATA_TYPE']

        if pd.api.types.is_numeric_dtype(property_value) and pd.notnull(property_uom) and data_type_rules == 'NUMERIC':
            complete_entries += 1
        elif pd.api.types.is_string_dtype(property_value) and pd.isnull(property_uom) and data_type_rules == 'STRING':
            complete_entries += 1

    completeness = (complete_entries / len(data)) * 100 if len(data) > 0 else 0
    return completeness

# Function to send email with attachment
def send_email(to_email, subject, body, attachment_path):
    from_email = "MantshXS@eskom.co.za" # Replace with your email
    from_password = "fancy=Koala" # Replace with your email password

    try:
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        with open(attachment_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename= {attachment_path}")
        msg.attach(part)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        st.sidebar.success("Email sent successfully!")

    except smtplib.SMTPAuthenticationError:
        st.sidebar.error("Failed to send email: Authentication error. Check your email credentials.")
    except smtplib.SMTPException as e:
        st.sidebar.error(f"Failed to send email: {e}")

# Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ['Changes', 'Creations'])

if page == 'Changes':
    st.title("GX Master Data Quality Tool")
    st.title("Changes")
else:
    st.title("GX Master Data Quality Tool")
    st.title("Creations")

# Sidebar for file upload or SQL connection
st.sidebar.title("Upload Data")
data_source = st.sidebar.radio("Choose Data Source", ('Upload CSV', 'SQL Database'))

df = None

if data_source == 'Upload CSV':
    file_upload = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if file_upload is not None:
        df = pd.read_csv(file_upload)
    else:
        st.sidebar.warning("Please upload a CSV file to proceed.")
        df = pd.read_csv("CREATION_RESTRUCTURED_SAMPLE.csv")
        st.stop()
else:
    server = st.sidebar.text_input("SQL Server")
    database = st.sidebar.text_input("Database")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    query = st.sidebar.text_area("SQL Query")
    if st.sidebar.button("Fetch Data"):
        try:
            conn = pymssql.connect(server=server, user=username, password=password, database=database)
            df = pd.read_sql_query(query, conn)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.stop()
if df is not None:
    # Perform validations and calculate stats
    validation_checks = {
        "CORP_NO_type_check": (validate_column_type, 'CORP_NO', int),
        "ERP_NO_type_check": (validate_column_type, 'ERP_NUMBER', int),
        "DESCR_type_check": (validate_column_type, 'DESCRIPTOR', str),
        "PROPERTY_TERM_type_check": (validate_column_type, 'PROPERTY_TERM', str),
        "PROPERTY_VALUE_check": (lambda df, column, regex: df[column].apply(lambda x: isinstance(x, numbers.Number) or isinstance(x, str)), 'PROPERTY_VALUE', None),
        "POD_check": (validate_column_regex, 'PURCHASE_ORDER_DESCRIPTION', '^[a-zA-Z\s]$|^NULL$'),
        "PROP_FFT_check": (validate_column_regex, 'PROP_FFT', '^[a-zA-Z\s]$|^NULL$'),
        "PROPERTY_UOM_type_check": (validate_column_type, 'PROPERTY_UOM', str),
        "DATA_TYPE_type_check": (validate_column_type, 'DATA_TYPE', str),
        "ORIGINATING_PLANT_TRM_type_check": (validate_column_type, 'PLANT_NAME', str),
        "ORIGINATING_DIVISION_type_check": (validate_column_type, 'ORIGINATING_DIVISION', str),
        "PLANT_GROUP_type_check": (validate_column_type, 'PLANT_GROUP', str),
    }

    validation_results = {}
    passed_counts = {}
    failed_counts = {}
    total_rows = len(df)
    for check, (func, column, rule) in validation_checks.items():
        validation_results[check] = func(df, column, rule)
        passed_counts[check] = validation_results[check].sum()
        failed_counts[check] = len(df) - passed_counts[check]

    passed_percentage = {check: (count / total_rows) * 100 for check, count in passed_counts.items()}
    failed_percentage = {check: (count / total_rows) * 100 for check, count in failed_counts.items()}

    total_validation_percentage = calculate_total_validation_percentage(validation_results)
    pod_duplication_percentage = calculate_duplication_percentage(df, 'PURCHASE_ORDER_DESCRIPTION')
    completeness_percentage = calculate_completeness(df)

    # Collect failed data points for each validation check
    failed_data_points = df.copy()
    for check, result in validation_results.items():
        failed_data_points[check] = ~result

    failed_data_points['Failed_Checks'] = failed_data_points.apply(
        lambda row: [check for check in validation_checks.keys() if row[check]], axis=1
    )

    failed_data_points = failed_data_points[failed_data_points['Failed_Checks'].map(len) > 0]

    valid_data_points = df[~df.index.isin(failed_data_points.index)]

    # Layout: Three main sections
    st.header("Statistics")

    # Gauge Charts for Statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_gauge_validation = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_validation_percentage,
            title={'text': "Total Validation Percentage"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        fig_gauge_validation.update_layout(width=300, height=300)
        st.plotly_chart(fig_gauge_validation)

        st.markdown("<br>", unsafe_allow_html=True) # Add space between gauges

    with col2:
        fig_gauge_pod = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pod_duplication_percentage,
            title={'text': "POD Duplication Percentage"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        fig_gauge_pod.update_layout(width=300, height=300)
        st.plotly_chart(fig_gauge_pod)

        st.markdown("<br>", unsafe_allow_html=True) # Add space between gauges

    with col3:
        fig_gauge_completeness = go.Figure(go.Indicator(
            mode="gauge+number",
            value=completeness_percentage,
            title={'text': "Completeness Percentage"},
            gauge={'axis': {'range': [None, 100]}}
        ))
        fig_gauge_completeness.update_layout(width=300, height=300)
        st.plotly_chart(fig_gauge_completeness)

    # Bar Chart of Validation Results
    labels = list(passed_percentage.keys())
    passed = list(passed_percentage.values())
    failed = list(failed_percentage.values())

    fig_validation = go.Figure()
    fig_validation.add_trace(go.Bar(
        y=labels,
        x=passed,
        name='Passed',
        orientation='h',
        marker_color='green'
    ))
    fig_validation.add_trace(go.Bar(
        y=labels,
        x=failed,
        name='Failed',
        orientation='h',
        marker_color='red'
    ))

    fig_validation.update_layout(barmode='stack', title="Validation Results", xaxis_title="Percentage", yaxis_title="Validation Check")
    st.plotly_chart(fig_validation)

    # Show validation results as tables
    st.header("Validation Results")
    st.write("Number of Items Processed :",total_rows)
    st.subheader("Passed Validation Checks")
    passed_df = pd.DataFrame({
        "Validation Check": list(passed_percentage.keys()),
        "Passed Percentage": list(passed_percentage.values())
    })
    st.write("Valid Data count : ", len(valid_data_points))
    st.dataframe(valid_data_points)
    st.subheader("Failed Validation Checks")
    failed_df = pd.DataFrame({
        "Validation Check": list(failed_percentage.keys()),
        "Failed Percentage": list(failed_percentage.values())
    })

    st.dataframe(failed_data_points)
    st.write("Invalid Data count : ", len(failed_data_points))
    # New visual: Number of failed vs passed values by day, week, month
    st.header("Failed vs Passed Over Time")
    # Function to convert date column to datetime if present
    def convert_to_datetime(df):
        date_columns = ['REQUEST_CREATION_DATE', 'CREATE_DATE']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                return col  # Return the name of the column converted to datetime
        return None  # Return None if neither column is found
    # Convert the appropriate column to datetime
    date_column = convert_to_datetime(df)
    
    if date_column is None:
        st.error("No valid date column found (either REQUEST_CREATION_DATE or CREATE_DATE).")
    else:
        # Extract date components based on selected timeframe
        df['CREATE_DAY'] = df[date_column].dt.date
        df['CREATE_WEEK'] = df[date_column].dt.to_period('W').astype(str)
        df['CREATE_MONTH'] = df[date_column].dt.to_period('M').astype(str)
        df['CREATE_HOUR'] = df[date_column].dt.strftime('%Y-%m-%d %H:00:00')
    
        timeframes = {
            'Hour': 'CREATE_HOUR',
            'Day': 'CREATE_DAY',
            'Week': 'CREATE_WEEK',
            'Month': 'CREATE_MONTH'
            
        }
    
        timeframe = st.selectbox('Select Timeframe', list(timeframes.keys()))
    
        # Assuming validation_results and validation_checks are defined somewhere
        # Calculate Total_Failed and Total_Passed
        failed_vs_passed = df.copy()
        for check, result in validation_results.items():
            failed_vs_passed[check] = result
    
        failed_vs_passed['Total_Failed'] = failed_vs_passed.apply(lambda row: sum(not row[check] for check in validation_checks.keys()), axis=1)
        failed_vs_passed['Total_Passed'] = len(validation_checks) - failed_vs_passed['Total_Failed']
    
        # Group by and aggregate without summing datetime columns
        grouped = failed_vs_passed.groupby(timeframes[timeframe]).agg({
            'Total_Failed': 'sum',
            'Total_Passed': 'sum'
        }).reset_index()
    
        # Plotting with Plotly Express
        fig_time = px.bar(grouped, x=timeframes[timeframe], y=['Total_Failed', 'Total_Passed'], barmode='stack', title=f"Failed vs Passed by {timeframe}")
        st.plotly_chart(fig_time)
    
    # Download failed data points as CSV
    st.header("Failed Data Points")
    st.dataframe(failed_data_points)
    csv = failed_data_points.to_csv(index=False)
    st.download_button(label="Download Failed Data Points as CSV", data=csv, file_name='failed_data_points.csv', mime='text/csv')

    # Filtered DataFrame by UPDATED_BY and counts
    st.header("Validation Counts by Uploader")
    uploader = st.selectbox('Select Uploader', df['UPDATED_BY'].unique())
    # Filtered DataFrame for the selected uploader
    filtered_df = df[df['UPDATED_BY'] == uploader]
    
    # Ensure 'Total_Failed' column exists in filtered_df
    if 'Total_Failed' not in filtered_df.columns:
        filtered_df['Total_Failed'] = 0
    
    # Calculate counts
    valid_count = len(filtered_df) - filtered_df['Total_Failed'].sum()
    invalid_count = filtered_df['Total_Failed'].sum()
    
    # Display filtered DataFrame
    st.write(filtered_df)
    
    # Display counts of valid and invalid entries
    st.write(f"Valid Items: {valid_count}")
    st.write(f"Invalid Items: {invalid_count}")

    # Email section
    st.sidebar.title("Send Report via Email")
    to_email = st.sidebar.text_input("Recipient Email")
    email_subject = st.sidebar.text_input("Email Subject")
    email_body = st.sidebar.text_area("Email Body")

    if st.sidebar.button("Send Email"):
        if not to_email or not email_subject or not email_body:
            st.sidebar.warning("Please fill in all the email fields.")
        else:
            try:
                # Save the CSV to a temporary file
                with tmp.NamedTemporaryFile(delete=False, suffix='.csv') as temp_csv:
                    temp_csv_path = temp_csv.name
                    temp_csv.write(csv.encode())

                send_email(to_email, email_subject, email_body, temp_csv_path)
            except Exception as e:
                st.sidebar.error(f"Error sending email: {e}")

else:
    st.warning("No data loaded. Please upload a CSV file or connect to an SQL database to proceed.")
