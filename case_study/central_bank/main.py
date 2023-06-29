import requests

url = "https://www.bis.org/review/r230515c.pdf"
filename = "r230515c.pdf"

response = requests.get(url)

if response.status_code == 200:
    with open(filename, 'wb') as file:
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
