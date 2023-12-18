import mlflow
import subprocess
from pyngrok import ngrok, conf
import getpass
import os
from dotenv import load_dotenv

load_dotenv()

# Acesse as variáveis de ambiente
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
subprocess.Popen(["mlflow", "ui", "--backend-store-uri", MLFLOW_TRACKING_URI])


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow will create an experiment if it doesn't exist
#mlflow.set_experiment("finetuning-ptt5-base")


print("Enter your authtoken, which can be copied from https://dashboard.ngrok.com/auth")
conf.get_default().auth_token = NGROK_AUTH_TOKEN  
port=5000
public_url = ngrok.connect(port).public_url
print(f' * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"')