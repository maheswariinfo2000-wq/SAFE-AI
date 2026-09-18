import smtplib
import pandas as pd
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

SENDER = os.getenv("EMAIL_SENDER")
APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
RECEIVER = os.getenv("EMAIL_RECEIVER")

DATA_PATH = "data/processed/hotspots_with_anomalies.csv"

def build_alert_body(anomalies_df):
    lines = [f"🚨 {len(anomalies_df)} thermal anomalies detected in the latest scan.\n"]
    for _, row in anomalies_df.iterrows():
        lines.append(
            f"- {row['nearest_industrial_name']} ({row['nearest_industrial_type']})\n"
            f"  FRP: {row['frp']} | Date: {row['acq_date']} | "
            f"Distance: {row['nearest_industrial_km']:.2f} km\n"
        )
    lines.append("\nView live dashboard: http://127.0.0.1:5000/alerts")
    return "\n".join(lines)

def send_alert_email():
    df = pd.read_csv(DATA_PATH)
    anomalies = df[df["is_anomaly"] == True]

    if len(anomalies) == 0:
        print("No anomalies found — no email sent.")
        return

    body = build_alert_body(anomalies)

    msg = MIMEMultipart()
    msg["From"] = SENDER
    msg["To"] = RECEIVER
    msg["Subject"] = f"⚠️ Industrial Fire Alert — {len(anomalies)} anomalies detected"
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER, APP_PASSWORD)
        server.sendmail(SENDER, RECEIVER, msg.as_string())
        server.quit()
        print(f"Alert email sent successfully to {RECEIVER}!")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    send_alert_email()