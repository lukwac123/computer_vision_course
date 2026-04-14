from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import Request, urlopen

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


SEARCH_QUERIES = ["horse", "lion"]
DOWNLOAD_LIMIT = 3
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "downloads"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/135.0.0.0 Safari/537.36"
)


def create_driver():
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")

    try:
        return webdriver.Chrome(options=chrome_options)
    except WebDriverException:
        firefox_options = FirefoxOptions()
        firefox_options.add_argument("-headless")
        return webdriver.Firefox(options=firefox_options)


def accept_google_consent(driver):
    consent_selectors = [
        "//button[.//div[contains(., 'Accept all')]]",
        "//button[contains(., 'Accept all')]",
        "//button[contains(., 'I agree')]",
        "//button[contains(., 'Zaakceptuj wszystko')]",
        "//button[contains(., 'Akceptuj wszystko')]",
    ]

    for selector in consent_selectors:
        try:
            button = WebDriverWait(driver, 2).until(
                EC.element_to_be_clickable((By.XPATH, selector))
            )
            button.click()
            return
        except TimeoutException:
            continue


def build_search_url(query):
    return f"https://www.google.com/search?tbm=isch&q={quote_plus(query)}"


def build_bing_search_url(query):
    return f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC3"


def is_google_blocked(driver):
    return "/sorry/" in driver.current_url or "nietypowy ruch" in driver.page_source.lower()


def fetch_full_image_url(driver):
    candidates = driver.find_elements(By.CSS_SELECTOR, "img.sFlh5c, img.n3VNCb")
    for image in candidates:
        src = image.get_attribute("src") or ""
        if src.startswith("http") and "gstatic.com" not in src:
            return src
    return None


def collect_google_image_urls(driver, limit):
    image_urls = []
    visited_urls = set()
    thumbnail_index = 0

    while len(image_urls) < limit:
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf")
        if thumbnail_index >= len(thumbnails):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(1)
            thumbnails = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf")
            if thumbnail_index >= len(thumbnails):
                break

        thumbnail = thumbnails[thumbnail_index]
        thumbnail_index += 1

        try:
            driver.execute_script("arguments[0].click();", thumbnail)
            WebDriverWait(driver, 5).until(lambda current_driver: fetch_full_image_url(current_driver))
            image_url = fetch_full_image_url(driver)
            if not image_url or image_url in visited_urls:
                continue

            visited_urls.add(image_url)
            image_urls.append(image_url)
        except (TimeoutException, WebDriverException):
            continue

    return image_urls


def extract_bing_image_urls(driver, limit):
    image_urls = []
    for result in driver.find_elements(By.CSS_SELECTOR, "a.iusc, a[href*='mediaurl=']"):
        href = result.get_attribute("href") or ""
        if "mediaurl=" not in href:
            continue

        parsed_url = urlparse(href)
        media_urls = parse_qs(parsed_url.query).get("mediaurl", [])
        if not media_urls:
            continue

        image_url = media_urls[0]
        if image_url.startswith("http") and image_url not in image_urls:
            image_urls.append(image_url)

        if len(image_urls) >= limit:
            break

    return image_urls


def infer_extension(image_url):
    path = urlparse(image_url).path.lower()
    for extension in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(extension):
            return extension
    return ".jpg"


def download_file(image_url, destination):
    request = Request(image_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=20) as response:
        destination.write_bytes(response.read())


def download_images(driver, query, limit):
    target_directory = OUTPUT_DIR / query.replace(" ", "_")
    target_directory.mkdir(parents=True, exist_ok=True)

    driver.get(build_search_url(query))
    accept_google_consent(driver)

    if is_google_blocked(driver):
        print(f"Google blocked automated access for '{query}', switching to Bing Images.")
        driver.get(build_bing_search_url(query))
        WebDriverWait(driver, 10).until(
            lambda current_driver: len(current_driver.find_elements(By.CSS_SELECTOR, "a[href*='mediaurl=']")) > 0
        )
        image_urls = extract_bing_image_urls(driver, limit)
    else:
        image_urls = collect_google_image_urls(driver, limit)

    downloaded = 0
    for image_url in image_urls:
        try:
            image_path = target_directory / f"{downloaded + 1:02d}{infer_extension(image_url)}"
            download_file(image_url, image_path)
            downloaded += 1
            print(f"Downloaded {query}: {image_path.name}")
        except (HTTPError, URLError, OSError, WebDriverException) as error:
            print(f"Skipped image for {query}: {error}")

    print(f"Finished {query}: {downloaded}/{limit} images")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    driver = create_driver()

    try:
        for query in SEARCH_QUERIES:
            download_images(driver, query, DOWNLOAD_LIMIT)
            print()
    finally:
        driver.quit()


if __name__ == "__main__":
    main()