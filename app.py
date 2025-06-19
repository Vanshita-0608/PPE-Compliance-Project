import streamlit as st
import pandas as pd
import os
from ultralytics import YOLO
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# === Risk update logic ===
def update_risk_excel_incrementally(label_dir, output_excel):
    class_names = {
        0: 'person', 1: 'glasses', 2: 'safety-vest',
        3: 'helmet', 4: 'gloves', 5: 'shoes'
    }

    if os.path.exists(output_excel):
        df_existing = pd.read_excel(output_excel)
        existing_images = set(df_existing['image'])
    else:
        df_existing = pd.DataFrame()
        existing_images = set()

    data = []
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            image_name = file.replace('.txt', '.jpg')
            if image_name in existing_images:
                continue

            with open(os.path.join(label_dir, file), 'r') as f:
                lines = f.readlines()
            class_ids = sorted(set(int(line.split()[0]) for line in lines if line.strip()))
            detected_classes = [class_names[i] for i in class_ids if i in class_names]
            class_ids_str = ', '.join(map(str, class_ids))

            if 0 not in class_ids:
                risk = 'No Risk'
            elif 3 not in class_ids and 2 not in class_ids:
                risk = 'High Risk'
            elif 3 not in class_ids or 2 not in class_ids:
                risk = 'Medium Risk'
            else:
                risk = 'Low Risk'

            data.append({
                'image': image_name,
                'class_ids': class_ids_str,
                'Detected Classes': ', '.join(detected_classes),
                'num_unique_classes': len(class_ids),
                'Risk Level': risk
            })

    df_new = pd.DataFrame(data)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_excel(output_excel, index=False)


# === YOLO prediction ===
def run_yolo_prediction(model_path, image_path):
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True, save_txt=True, conf=0.1)
    label_dirs = sorted(glob('runs/detect/*/labels'), key=os.path.getmtime)
    return label_dirs[-1] if label_dirs else None


# === Risk visualization ===
def show_risk_charts(df):
    st.header("📊 PPE Compliance Risk Analysis")
    risk_counts = df['Risk Level'].value_counts()
    total_images = len(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Images", total_images)
    col2.metric("High Risk", risk_counts.get('High Risk', 0))
    col3.metric("Medium Risk", risk_counts.get('Medium Risk', 0))

    st.markdown("---")
    st.subheader("Risk Level Distribution")
    fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=120)
    colors = sns.color_palette("Set2", len(risk_counts))
    ax1.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    ax1.axis('equal')
    st.pyplot(fig1)

    st.markdown("---")
    st.subheader("Detected PPE Classes Frequency")
    all_classes = df['Detected Classes'].dropna().str.split(', ').explode()
    class_freq = all_classes.value_counts()

    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=120)
    sns.barplot(x=class_freq.index, y=class_freq.values, palette="viridis", ax=ax2)
    ax2.set_xlabel("PPE Class")
    ax2.set_ylabel("Count")
    ax2.set_title("Detected PPE Class Counts")
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig2)


# === Streamlit app ===
def main():
    st.set_page_config(page_title="PPE Safety Dashboard", layout="wide")
    st.title("🦺 PPE Compliance & Risk Analysis")

   # Hidden, fixed paths for model and Excel file
    model_path = r"C:\Users\DELL\OneDrive\Desktop\Worker_Safety\best (1).pt"
    excel_path = r"C:\Users\DELL\OneDrive\Desktop\Worker_Safety\prediction_summary.xlsx"

    uploaded_file = st.file_uploader("Upload Worker Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        img_save_path = os.path.join("uploaded_images", uploaded_file.name)
        os.makedirs("uploaded_images", exist_ok=True)
        with open(img_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.image(img_save_path, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Prediction & Update Report"):
            with st.spinner("Running YOLO prediction..."):
                label_dir = run_yolo_prediction(model_path, img_save_path)
                if label_dir:
                    update_risk_excel_incrementally(label_dir, excel_path)
                    st.success("Report updated successfully!")
                else:
                    st.error("Prediction failed. No label directory found.")

    # Navigation buttons to other sections
    if os.path.exists(excel_path):
        if st.button("📄 View Updated Excel File"):
            df = pd.read_excel(excel_path)
            st.dataframe(df)

        if st.button("📊 View Charts"):
            df = pd.read_excel(excel_path)
            show_risk_charts(df)


if __name__ == "__main__":
    main()

