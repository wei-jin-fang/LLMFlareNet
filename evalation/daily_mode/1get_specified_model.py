import pandas as pd
import os

# Read the CSV file
df = pd.read_csv('../../scripts/get_daily_data/result_forecast.csv')  # Replace with your CSV file path if different
# Modify Class column to keep only the first character
df['Class'] = df['Class'].str[0]
# Get unique model types
model_types = df['_modelType'].unique()
# Remove rows where CV_average is less than 0
df = df[df['CV_average'] >= 0]
# Create output directory if it doesn't exist
output_dir = 'split_result_csvs'
os.makedirs(output_dir, exist_ok=True)

# Split and save CSV for each model type
for model in model_types:
    # Filter data for current model
    model_df = df[df['_modelType'] == model]

    # Create output filename
    output_file = os.path.join(output_dir, f'{model}.csv')

    # Save to CSV
    model_df.to_csv(output_file, index=False)
    print(f'Saved {output_file}')