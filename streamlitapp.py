import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import streamlit_authenticator as stauth
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.io as pio
import tempfile
import os

# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="FATUMAYAN", page_icon=":guardsman:", layout="wide")

# Hashed passwords generated using Hasher(['password1', 'password2', ...]).generate()
hashed_passwords = stauth.Hasher(['password123', 'password456', 'password789']).generate()

credentials = {
    "usernames": {
        "dennis": {
            "name": "Dennis",
            "password": hashed_passwords[0],
            "role": "admin"
        },
        "john": {
            "name": "John",
            "password": hashed_passwords[1],
            "role": "doctor"
        },
        "mary": {
            "name": "Mary",
            "password": hashed_passwords[2],
            "role": "nurse"
        }
    }
}
authenticator = stauth.Authenticate(
    credentials,
    "cancer_dashboard",       # Cookie name
    "auth_token",             # Signature key
    cookie_expiry_days=1
)

name, authentication_status, username = authenticator.login(fields={'Form name': 'Login', 'Username': 'Username', 'Password': 'Password'})

# Main logic
if authentication_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.write(f"👋 Welcome, {name}!")
    role = credentials["usernames"][username]["role"]
    st.sidebar.write(f"Role: {role}")

    # ---------- All app logic STARTS here after login ----------

    st.title("🧬 Cancer Patient Data Dashboard")

    uploaded_file = st.sidebar.file_uploader("Upload a new CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ New data uploaded successfully!")
    else:
        df = pd.read_csv("cancerpatients.csv")

    le_gender = LabelEncoder().fit(df['Gender'])
    le_cancer = LabelEncoder().fit(df['Cancer_Type'])

    df['Gender'] = le_gender.transform(df['Gender'])
    df['Cancer_Type'] = le_cancer.transform(df['Cancer_Type'])

    features = ['Age', 'Obesity_Level', 'Gender', 'Cancer_Type']
    X = df[features]
    y = df['Survival_Years']
    rf_model = RandomForestRegressor().fit(X, y)

    selected_country = st.sidebar.multiselect("Country", options=df["Country_Region"].unique(), default=df["Country_Region"].unique())
    selected_gender = st.sidebar.multiselect("Gender", options=le_gender.classes_, default=le_gender.classes_)
    selected_cancer = st.sidebar.multiselect("Cancer Type", options=le_cancer.classes_, default=le_cancer.classes_)
    patient_id = st.sidebar.text_input("Search Patient ID")

    filtered_df = df[
        df["Country_Region"].isin(selected_country) &
        df["Gender"].isin(le_gender.transform(selected_gender)) &
        df["Cancer_Type"].isin(le_cancer.transform(selected_cancer))
    ]

    if patient_id:
        filtered_df = filtered_df[filtered_df["Patient ID"].astype(str).str.contains(patient_id)]

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🌍 Country Trends", "📈 Custom Analysis"])

    with tab1:
        st.subheader("Age Distribution by Gender")
        fig = px.histogram(filtered_df, x='Age', color='Gender', barmode='group')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Cancer Type by Country")
        fig2 = px.scatter(filtered_df, x='Country_Region', y='Cancer_Type', color='Gender', opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 Summary Report")
        avg_survival = filtered_df["Survival_Years"].mean()
        most_common_cancer_code = filtered_df["Cancer_Type"].mode()[0]
        most_common_cancer = le_cancer.inverse_transform([most_common_cancer_code])[0]
        total_patients = len(filtered_df)

        st.markdown(f"""
        - **Average Survival Years:** `{avg_survival:.2f}` years  
        - **Most Common Cancer Type:** `{most_common_cancer}`  
        - **Total Patients in Selection:** `{total_patients}`
        """)

    with tab2:
        st.subheader("Survival Years by Country")
        fig3 = px.box(filtered_df, x='Country_Region', y='Survival_Years', color='Country_Region')
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Obesity Level vs Country")
        fig4 = px.strip(filtered_df, x='Country_Region', y='Obesity_Level', color='Gender', stripmode='overlay')
        st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.subheader("Customize Your Chart")
        x_axis = st.selectbox("Select X-axis", options=df.columns)
        y_axis = st.selectbox("Select Y-axis", options=df.columns)
        chart_type = st.radio("Chart Type", options=["Line", "Scatter", "Bar"])

        if chart_type == "Scatter":
            fig5 = px.scatter(filtered_df, x=x_axis, y=y_axis, color='Gender')
        elif chart_type == "Line":
            fig5 = px.line(filtered_df, x=x_axis, y=y_axis, color='Gender')
        else:
            fig5 = px.bar(filtered_df, x=x_axis, y=y_axis, color='Gender')

        st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Predict Survival Years for Hypothetical Scenario")
    hypothetical_age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
    hypothetical_obesity = st.number_input("Enter Obesity Level (0-10)", min_value=0, max_value=10, value=5)
    hypothetical_gender = st.selectbox("Select Gender", options=le_gender.classes_)
    hypothetical_cancer_type = st.selectbox("Select Cancer Type", options=le_cancer.classes_)

    try:
        hypothetical_features = pd.DataFrame([[
            hypothetical_age,
            hypothetical_obesity,
            le_gender.transform([hypothetical_gender])[0],
            le_cancer.transform([hypothetical_cancer_type])[0]
        ]], columns=features)
        prediction = rf_model.predict(hypothetical_features)
        st.success(f"✅ Predicted Survival Years: {prediction[0]:.2f}")
    except ValueError as e:
        st.error(f"Encoding error: {e}")

    # ---------- Export to PDF Section ----------
    st.subheader("🖨️ Export Chart to PDF")

    # Let user select chart to export
    export_chart = st.selectbox("Select chart to export:", ["Age Distribution", "Cancer Type by Country"])

    # Generate the figure based on user selection
    if export_chart == "Age Distribution":
        export_fig = px.histogram(filtered_df, x='Age', color='Gender', barmode='group')
    elif export_chart == "Cancer Type by Country":
        export_fig = px.scatter(filtered_df, x='Country_Region', y='Cancer_Type', color='Gender', opacity=0.6)

    # Export to PDF when the button is clicked
    if st.button("📤 Export to PDF"):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "plot.png")
            pdf_path = os.path.join(tmpdir, "report.pdf")

            # Save chart as image (requires kaleido)
            export_fig.write_image(image_path)

            # Create a PDF file and embed the image
            c = canvas.Canvas(pdf_path, pagesize=letter)
            c.drawString(100, 750, f"{export_chart} Report")
            c.drawImage(image_path, 50, 300, width=500, height=400)
            c.save()

            # Allow the user to download the generated PDF
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=f,
                    file_name=f"{export_chart.lower().replace(' ', '_')}_report.pdf"
                )

    st.sidebar.markdown("### Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="filtered_cancer_data.csv",
        mime="text/csv"
    )

    # ---------- All app logic ENDS here ----------

elif authentication_status is False:
    st.error("❌ Incorrect username or password")
elif authentication_status is None:
    st.warning("Please enter your credentials")
