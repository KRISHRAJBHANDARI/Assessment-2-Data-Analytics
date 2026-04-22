import pandas as pd

def process_fuel_data(file_path):
    # 1. Load Petrol and Diesel sheets
    petrol = pd.read_excel(file_path, sheet_name='Petrol TGP')
    diesel = pd.read_excel(file_path, sheet_name='Diesel TGP')

    # 2. Standardize Date Columns
    petrol.rename(columns={petrol.columns[0]: 'Date'}, inplace=True)
    diesel.rename(columns={diesel.columns[0]: 'Date'}, inplace=True)

    # 3. Extract National Average (handles the newline characters in headers)
    nat_petrol_col = [c for c in petrol.columns if 'National' in c][0]
    nat_diesel_col = [c for c in diesel.columns if 'National' in c][0]

    petrol_df = petrol[['Date', nat_petrol_col]].rename(columns={nat_petrol_col: 'petrol_price'})
    diesel_df = diesel[['Date', nat_diesel_col]].rename(columns={nat_diesel_col: 'diesel_price'})

    # 4. Convert to Datetime
    petrol_df['Date'] = pd.to_datetime(petrol_df['Date'])
    diesel_df['Date'] = pd.to_datetime(diesel_df['Date'])

    # 5. Resample to Monthly (matching your Opal data frequency)
    # We use 'mean' to get the average price people paid that month
    fuel_monthly = petrol_df.set_index('Date').resample('MS').mean()
    diesel_monthly = diesel_df.set_index('Date').resample('MS').mean()

    # 6. Combine and Save
    fuel_combined = fuel_monthly.join(diesel_monthly).reset_index()
    fuel_combined.to_csv('fuel_monthly_cleaned.csv', index=False)
    print("✓ Fuel data cleaned and saved to fuel_monthly_cleaned.csv")

if __name__ == "__main__":
    process_fuel_data('data/AIP_TGP_Data_17-Apr-2026.xlsx')