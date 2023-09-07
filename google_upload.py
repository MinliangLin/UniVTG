from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pathlib import Path

settings = {
    "client_config_backend": "service",
    "service_config": {
        "client_json_file_path": "/home/ubuntu/UniVTG/data/service-account.json"
    }
}
gauth = GoogleAuth(settings=settings)
gauth.ServiceAuth()
drive = GoogleDrive(gauth)

data = [
'data/data2022/PRrcjKBiXzeoX_ujGFJuWxEw-UWpqt8C/V-z3sTTk1Q7dKC@1651236546.mp4',
'data/data2022/2o3QMADs3lGjVsONduY-ZKm-hWO7bwP4/V-ZSYpC7pODe3x@1652012799.mp4',
'data/data2022/ChXut1zDZzAACy_fjPNzvQO0dtAxMVNn/V-JhPFJ27m2dmA@1650632234.mp4',
'data/data2022/2o3QMADs3lGjVsONduY-ZKm-hWO7bwP4/V-rmIBd26T7d1r@1651243023.mp4',
'data/data2022/y-hYCdlGZXs4fvwM-oiRYeFgrft5m7J6/V-fltQQTfBLeVv@1653026278.mp4',
'data/data2022/ld0ThOdGzUCJjBpFyJR7l0Xpcu2LaIgx/V-ZaITACQvodIE@1648827669.mp4',
'data/data2022/0rcPDXcxY1yhOW7APe9E4bKYfqzQEMeh/V-7XDG8mU1TeXR@1654176194.mp4',
'data/data2022/Z3L98X-3QxCqhcQKgiujvNej0-B49f7o/V-hn9G3eWg7dn9@1651269079.mp4',
'data/data2022/L_2wudwL7-WcRY6gnrbwgr21A7JEEXwT/V-97b0I6Hmqd1g@1649068786.mp4',
'data/data2022/pbPt-oj8vIGXs-UeQSfKK4FHqgKNhI_1/V-EAoeQoBYOe8P@1653463975.mp4',
'data/data2022/d1-LgdLhHi11A0JUocASz-xPAK-tL1hL/V-XOm1z935yekc@1658220154.mp4',
'data/data2022/q16wr-P4YiFomArgR9xp_22x8euXFRlE/V-exqEzwhyNevv@1653389497.mp4',
]

for x in data:
    print(x)
    f = str(x)
    fp = drive.CreateFile({'parents': [{'id': '1wHbfcmTgXl55aAZ2dN4FL32O8vMj4iN8'}]})
    fp.SetContentFile(f)
    fp.Upload()
