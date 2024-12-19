import pandas as pd

# Paths to the CSV files
file1 = r"C:\Users\pc\data_tools\dataset\Van_Zoocasa_HS_Single_Listing1.csv"
file2 = r"C:\Users\pc\data_tools\dataset\Van_Zoocasa_HS_Single_Listing2.csv"
file3 = r"C:\Users\pc\data_tools\dataset\Van_Zoocasa_HS_Single_Listing3.csv"
file4 = r"C:\Users\pc\data_tools\dataset\Van_Zoocasa_HS_Single_Listing4.csv"
file5 = r"C:\Users\pc\data_tools\dataset\Van_Zoocasa_HS_Single_Listing5.csv"


# Load the CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
df5 = pd.read_csv(file5)

# Merge the DataFrames
merged_df = pd.concat([df1, df2, df3,df4,df5], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("house_file.csv", index=False)

print("Files merged successfully!")

duplicates_count = merged_df.duplicated().sum()
print(merged_df.count)
print(f"Number of duplicate rows: {duplicates_count}")







