from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time






# Selenium setup
options = Options()
options.add_argument("--headless")  # Run in headless mode
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")

# Path to the ChromeDriver (update with the correct path on your machine)
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

df = pd.read_csv(r"C:\Users\pc\house_urls_unique5.csv")  # Replace with the path to your CSV file
pages = df.astype(str).values.flatten().tolist()

post_types = []
post_sizes = []
post_levels = []
post_garages = []
taxes_list = []
post_mlsnumbers = []
post_prices = []
post_links = []
post_address=[]
garage_=[]
bathroom_=[]
bedroom_=[]
date_=[]

for page in pages:
    print(f"Scraping page: {page}")
    try:
        # Load the page in Selenium
        driver.get(page)
        time.sleep(5)  # Wait for the page to load fully

        # Extract price using full XPath
        try:
            price_element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@class='style_price__C0jEc' and @data-testid='listingPriceModal']")))
            post_price = price_element.text.strip()
        except Exception as e:
            print(f"Error extracting price: {e}")
            post_price = "N/A"


        # Extract other attributes (update these XPaths as needed)
        try:
            type_element = driver.find_element(By.XPATH, "//span[@data-testid='TypeKeyFacts']")
            post_type = type_element.text.strip() if type_element else "N/A"
        except Exception:
            post_type = "N/A"

        try:
            size_element = driver.find_element(By.XPATH, "//span[@data-testid='SizeKeyFacts']")
            post_size = size_element.text.strip() if size_element else "N/A"
        except Exception:
            post_size = "N/A"

        try:
            level_element = driver.find_element(By.XPATH, "//span[@data-testid='LevelsKeyFacts']")
            post_level = level_element.text.strip() if level_element else "N/A"
        except Exception:
            post_level = "N/A"

        try:
            garage_element = driver.find_element(By.XPATH, "//span[@data-testid='GarageKeyFacts']")
            post_garage = garage_element.text.strip() if garage_element else "N/A"
        except Exception:
            post_garage = "N/A"

        try:
            taxes_element = driver.find_element(By.XPATH, "//span[@data-testid='TaxesKeyFacts']")
            post_taxes = taxes_element.text.strip() if taxes_element else "N/A"
        except Exception:
            post_taxes = "N/A"

        try:
            mls_element = driver.find_element(By.XPATH, "//span[@data-testid='MLS®NumberKeyFacts']")
            post_mlsnumber = mls_element.text.strip() if mls_element else "N/A"
        except Exception:
            post_mlsnumber = "N/A"

        try:
            # Locate the "Property Address" element using its class and data-testid attribute
            property_address_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//p[@class='style_address__ImkKk' and @data-testid='keyFactsPropertyAddress']"))
            )
            property_address = property_address_element.text.replace("Property Address:", "")  # Extract and clean text
        except Exception as e:
            print(f"Error extracting Property Address: {e}")
            property_address = "N/A"  # Default value if the element is not found

        try:
            # Locate the element using its class and data-testid attributes
            garage_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//span[@class='style_stat__rJKvu' and @data-testid='listingCarIcon']"))
            )
            # Extract and clean the text content (the number following the </svg>)
            garage_count = garage_element.text.strip()
        except Exception as e:
            print(f"Error extracting garage count: {e}")
            garage_count = "N/A"  # Default value if element not found



        try:
            # Locate the element using its class and data-testid attributes
            bedroom_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//span[@class='style_stat__rJKvu' and @data-testid='listingBedIcon']"))
            )
            # Extract and clean the text content (the number following the </svg>)
            bedroom_count = bedroom_element.text.strip()
        except Exception as e:
            print(f"Error extracting bed room count: {e}")
            bedroom_count = "N/A"  # Default value if element not found



        try:
            # Locate the element using its class and data-testid attributes
            bathroom_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//span[@class='style_stat__rJKvu' and @data-testid='listingBathIcon']"))
            )
            # Extract and clean the text content (the number following the </svg>)
            bathroom_count = bathroom_element.text.strip()
        except Exception as e:
            print(f"Error extracting bathroom count: {e}")
            bathroom_count = "N/A"  # Default value if element not found

        try:
            # Wait for the element to be present and scrape it
            age_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//span[@data-testid='Approx.AgeKeyFacts']"))
            )
            approx_age = age_element.text.strip()
        except Exception as e:
            print(f"Error scraping Approximate Age: {e}")
            approx_age = "N/A"  # Default value if the element is not found


        # Append data to lists
        post_prices.append(post_price)
        post_types.append(post_type)
        post_sizes.append(post_size)
        post_levels.append(post_level)
        post_garages.append(post_garage)
        taxes_list.append(post_taxes)
        post_mlsnumbers.append(post_mlsnumber)
        post_links.append(page)
        post_address.append(property_address)
        garage_.append(garage_count)
        bathroom_.append(bathroom_count)
        bedroom_.append(bedroom_count)
        date_.append(approx_age)

    except Exception as e:
        print(f"Error scraping page {page}: {e}")
        continue

# Close the Selenium driver
driver.quit()

# Create DataFrame with all attributes
Van_sales = pd.DataFrame({
    'URL': post_links,
    'Type': post_types,
    'Size': post_sizes,
    'Levels': post_levels,
    'Garage': post_garages,
    'Taxes': taxes_list,
    'MLS Number': post_mlsnumbers,
    'Price': post_prices,
    'address':post_address,
    'garage':garage_,
    'bedroom':bedroom_,
    'bathroom':bathroom_,
    'date released':date_,
})

# Save to CSV
Van_sales.to_csv("Van_Zoocasa_HS_Single_Listing5.csv", index=False)
print("Scraping completed successfully!")
