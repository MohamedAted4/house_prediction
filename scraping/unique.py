import pandas as pd
from collections import Counter

# Load the CSV file
df = pd.read_csv(r"C:\Users\pc\house_urls5.csv")

# Convert the entire DataFrame to a list of strings
pages = df.astype(str).values.flatten().tolist()

# Count occurrences of each URL
url_counts = Counter(pages)

# Filter out URLs that occur more than once (duplicates)
duplicated_urls = {url: count for url, count in url_counts.items() if count > 1}

# Remove duplicates by keeping only unique URLs
unique_urls = list(set(pages))

# Create a DataFrame with unique URLs
df_unique = pd.DataFrame(unique_urls, columns=['House URLs'])

# Save the cleaned DataFrame to a new CSV file
df_unique.to_csv(r"C:\Users\pc\house_urls_unique5.csv", index=False)

print("Duplicates removed and new CSV file created.")
