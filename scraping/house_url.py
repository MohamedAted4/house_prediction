import requests
from bs4 import BeautifulSoup
import pandas as pd

# List of URLs to scrape
url_list = [
    "https://www.zoocasa.com/vancouver-bc-real-estate?page=1",
    "https://www.zoocasa.com/vancouver-bc-real-estate?page=2",
    "https://www.zoocasa.com/vancouver-bc-real-estate?page=3",
    "https://www.zoocasa.com/vancouver-bc-real-estate?page=4",
    "https://www.zoocasa.com/vancouver-bc-real-estate?page=5",

    
]
all_house_urls = []

for page in url_list:
    # Fetch the content of the page
    response = requests.get(page)

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the anchor tags that link to house listings
    listing_cards = soup.find_all('a', href=True)

    # Extract the URLs that are likely to be house listings
    for card in listing_cards:
        href = card['href']

        # Filter out URLs with query parameters like "?page=" or "?status="
        if '/vancouver-bc-real-estate/' in href and not any(param in href for param in ['?page=', '?status=']):
            # Check if it's a relative URL
            if href.startswith('/'):
                full_url = 'https://www.zoocasa.com' + href  # Concatenate the base URL
                all_house_urls.append(full_url)
            else:
                all_house_urls.append(href)  # If it's an absolute URL, just append it

# Create a DataFrame to store the URLs
df = pd.DataFrame(all_house_urls, columns=['House URLs'])

# Save the DataFrame to a CSV file
df.to_csv('house_urls5.csv', index=False)

# Output the DataFrame
print(df)
