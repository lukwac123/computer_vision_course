from google_images_download import google_images_download


response = google_images_download.googleimagesdownload()

search_querries = ['horse', 'lion']

def download_images(query):
    arguments = {
        'keywords': query,
        'format': 'jpg',
        'limit': 3,
        'print_urls': True,
    }
    try:
        response.download(arguments)
    except Exception as e:
        print(e)


for query in search_querries:
    download_images(query)
    print()