from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

settings = {
    "client_config_backend": "service",
    "service_config": {
        "client_json_file_path": "./data/service-account.json"
    }
}
gauth = GoogleAuth(settings=settings)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)

# file1 = drive.CreateFile({'id': '1gy8wKqA9gcYbk3tHewXX5qZ9SQAFhk6J'})
# file1.GetContentFile('results/model_best_pt_ft.ckpt')

# file2 = drive.CreateFile({'id': '1YKxqRTi19OO1CYgLe7NS71agpjQz40cW'})
# file2.GetContentFile('results/opt.json')

file1 = drive.CreateFile({'id': '1a4zLvaiDBr-36pasffmgpvH5P7CKmpze'})
file1.GetContentFile('pretrained_minigpt4.pth')
