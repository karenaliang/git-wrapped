# from flask import Flask, request, jsonify
# import requests

from flask import Flask

app = Flask(__name__)

@app.route('/hello')
def hello():
    return 'Hello, World!'


if __name__ == '__main__':
    app.run()


# Replace with your app details
# APP_ID = '839724'
# PRIVATE_KEY = """
# -----BEGIN RSA PRIVATE KEY-----
# MIIEpAIBAAKCAQEA0nGBoeNYYMKhBrJPhgDUFSw60J9EXIrVnZvjHlGA+OImwgUk
# y7klw0GSTeU+pu3mE9bJAfgvq+aqPnt5mQVmiZ5XHF9UmRuTgbpmk6WBcMV3eqYW
# hPeXXyCInudEnbmWToE9xkEM91wYqKIYmvfSfuQbqBpBq1SrkFS0kWGAPC8XLVbi
# RoUgcTMr0utFoaKgDbo6ykhkh0LvCg4bbrG+HdyW5Uq5pq+BvVZWBfZAZYndUm53
# tS7BBfTpWjWMKNuN6wLN8yZaW802LKPrIl4tE9AEhkWrxhAc/U7BNn3J7MOsrCcw
# LTCPAXdZl2WYik6uRDTiXS+7zYwnXsyW2SDd2wIDAQABAoIBAQCQup98xu4xHanB
# AUDP2CIDrbeFYwOQ4aqoCl3YrBUXFfGx3ffAZEUkVCRajh7YjyR84Gq9gALJJopP
# DnxCUQSEAPHIAYgdBGod/iQtxtWOyT+yiidSqzTfp3BOWJ9IGirVMu1ZO8O/Gwea
# cmCZs3aA7kBXDDe9rS2QxbCpi3S17iz/yb8bKWuQRlGGIQNmlf90bqiRMlNQakre
# RYqVI+9sqG5Vs4be28wfJtcSfmUYEk2OJnUL9dHviECvok40LkWCOWNot1M43viw
# KaEPDzmqiEo9QCzujAITOHpyiPkPuVxOhK23Kej4So2Uoq/BkrFKoupyQnyrDzM4
# jvfjWuEBAoGBAPfZ9I9uaNnfryc3I8b6ct570Jul+tOQqfKblW3brX0vEU/dl2qx
# Fe2/xLF0z6kvaW6jyTolVpPfY2yTdkPxnWCZm/3LEyTDsezVTIY7dWWtdEHBVEN3
# aaxjYIka1T2IOkqfuuqdXGNI73o+hQXc0oHWoy7ghDbheUtk9Cue3ufDAoGBANlc
# tMFSkGDG2YXbA9+qliPW5hgwyDlu7p5qj7Xf0OHfTOVwYUQhqPb76pNt3AdREzkn
# N1osotf3+0RMC+dN5CdofZdYuVrxhFTL7FWV2HZfiBTj9rYa4/W9fot4bJg/iTlK
# fHgc6sxg584VzwJADjJCYPZXyprtpFzi11NnlugJAoGARK6oX8rX0XB0CCj2iCBF
# DrQ+5bMEI/aVsb49lXjnxaXyZTBh0lYKXpzis85L8XVLATbv/2XzbPzdf2wJvOBK
# nUdT++t3fZuhRaECGLyHVsrPSZ16bUu9A+FyEexWpTuH87/5uyQqQvUmL7j9gddk
# mkhRZh8fZFntE+CA7UmUOaMCgYA02291R5+meRSykVT60h/arUqc/VfjZ4+NZHp4
# Dupb0xJ/BPoSOGE/VnlvyppkCRo0ns2+FvausDYhIKEQaYee8bEA2emRLQQyHrjl
# AyB1gLu14M3A8P8YZjFctzcpOuIi1XM5Pkb21mXoXrNNZ8hL/opXvXeef0Wl9N/J
# /3naCQKBgQCKEFVLpi64fnl57pwH4L0H78BU3Vm5Nn8UXkM3I5XenXRY4Po3pCcK
# ynr4XR5KcYTS9OIwfBt7O15vZwUqnufYZnntlzeOJcol7bHa2yaz6S14iCNlP7xu
# 69P2LgxqUWB2BZ3nNrWziTALgWeK9eDJkmBasgtpz+7E8oRmST1A4A==
# -----END RSA PRIVATE KEY-----
# """

# # Endpoint to handle installation events
# @app.route('/install', methods=['POST'])
# def handle_installation():
#     installation_payload = request.get_json()

#     # Authenticate and get an installation access token
#     installation_id = installation_payload['installation']['id']
#     token_url = f'https://api.github.com/app/installations/{installation_id}/access_tokens'
#     response = requests.post(token_url, headers={'Authorization': f'Bearer {PRIVATE_KEY}'})
#     access_token = response.json()['token']

#     # TODO: Retrieve repository information and statistics here

#     return jsonify({'message': 'Installation handled successfully'})

# if __name__ == '__main__':
#     app.run(debug=True)

