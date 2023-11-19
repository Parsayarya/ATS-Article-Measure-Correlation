import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_document(document_number, base_url):
    """
    Scrape data from a specific document page.

    Parameters:
    document_number (int): The document number to scrape.
    base_url (str): The base URL of the website.

    Returns:
    dict: A dictionary containing the scraped data.
    """
    url = f"{base_url}{document_number}"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.HTTPError as e:
        print(f"Failed to retrieve page {document_number}: {e}")
        return None
    except requests.RequestException as e:
        print(f"A request error occurred: {e}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    elements = soup.find_all('span', class_='characteristics__item__text__text')
    content_div = soup.find('div', class_='text-container')
    title = soup.find('h1', class_='title')

    data = {
        'Document_Number': document_number,
        'Subject': elements[0].get_text(strip=True) if len(elements) > 0 else None,
        'Status': elements[1].get_text(strip=True) if len(elements) > 1 else None,
        'Category': elements[2].get_text(strip=True) if len(elements) > 2 else None,
        'Topics': elements[3].get_text(strip=True) if len(elements) > 3 else None,
        'Title': title.get_text(strip=True) if title else None,
        'Content': content_div.get_text(strip=True) if content_div else None
    }

    return data

def main():
    """
    Main function to scrape data from a range of document numbers.
    """
    base_url = 'https://www.ats.aq/devAS/Meetings/Measure/'
    data = pd.DataFrame(columns=['Document_Number', 'Subject', 'Status', 'Category', 'Topics', 'Title', 'Content'])

    for i in range(808, 0, -1):
        scraped_data = scrape_document(i, base_url)
        if scraped_data:
            data = data.append(scraped_data, ignore_index=True)
        time.sleep(1)

    data.to_csv('scraped_content4.csv', index=False)

if __name__ == "__main__":
    main()
