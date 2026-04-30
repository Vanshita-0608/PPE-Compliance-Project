# PPE-Compliance-Project

# 🦺 PPE Compliance Monitoring System Using AI

🚀 End-to-end AI-powered PPE compliance monitoring system using YOLOv8, Streamlit, and Power BI for real-time detection, automated logging, and safety analytics.

## 📌 Project Overview

This project leverages deep learning and computer vision to detect whether individuals in a workspace are wearing mandatory Personal Protective Equipment (PPE), including:

- Person  
- Glasses  
- Safety Vest  
- Helmet  
- Gloves  
- Shoes  

It provides real-time detection, visual alerts, and automated data logging to improve workplace safety and compliance monitoring.


![Label ex](https://github.com/user-attachments/assets/dccd50d9-4e8a-4e4d-aea5-8b169fc2e370)


---

 📊 PPE Compliance Dashboard

![Dashboard](./Worker%20Safety%20Analysis%20Dashboard.jpg)
)

The dashboard provides insights into:

- PPE compliance trends  
- Risk classification (High / Medium / Low)  
- Frequent safety violations  

**Key Insight:** Majority of cases fall under high-risk category due to missing helmet and safety vest.
🛠️ Tech Stack

- Python  
- YOLOv8 (Computer Vision Model)  
- Streamlit (Web Interface)  
- Power BI (Dashboard & Analytics)  
- Excel (Data Logging)

- ⚙️ System Workflow

1. Input image/video is provided  
2. YOLOv8 model detects PPE components  
3. Detection results are logged in an Excel file  
4. Data is visualized in Power BI dashboard  
5. Insights help identify safety risks and trends  

 📊 Model Performance

- Detection Accuracy: ~80%  
- PPE Classes: 6 categories  
- Supports real-time inference via Streamlit  

🏗️ Real-World Applications

- Construction sites  
- Manufacturing industries  
- Warehouses and industrial zones  

This system helps safety teams monitor compliance and take proactive actions to reduce workplace risks.

---

## 📁 Folder Structure
PE-Compliance-Project/
├── app.py # Streamlit application
├── best.pt # Trained YOLOv8 model
├── classes.txt # PPE class labels
├── PPE.pbix # Power BI dashboard
├── prediction_summary.xlsx # Auto-generated predictions log
├── README.md # Project documentation
└── LICENSE # License file

