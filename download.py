from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pathlib import Path

settings = {
    "client_config_backend": "service",
    "service_config": {
        "client_json_file_path": "/home/ubuntu/minliang/data/service-account.json"
    }
}
gauth = GoogleAuth(settings=settings)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)

# file1 = drive.CreateFile({'id': '1gy8wKqA9gcYbk3tHewXX5qZ9SQAFhk6J'})
# file1.GetContentFile('model_best_pt_ft.ckpt')

file1 = drive.CreateFile({'id': '1YKxqRTi19OO1CYgLe7NS71agpjQz40cW'})
file1.GetContentFile('results/opt.json')
