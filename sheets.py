import gspread
from oauth2client.service_account import ServiceAccountCredentials

def log_results(results):
    # Replace these with your actual credentials file and sheet name
    CREDENTIALS_FILE = 'keys/tailcp-540f465346d0.json'
    SHEET_NAME = 'TailCP Results'

    # Scope for accessing Google Sheets API
    SCOPE = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    # Authenticate with Google Sheets API using credentials
    credentials = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, SCOPE)
    client = gspread.authorize(credentials)

    # Open the specified Google Sheet
    sheet = client.open(SHEET_NAME).sheet1  # You might need to adjust the sheet index (e.g., sheet2)

    # Data to log

    # Append the data to the sheet
    sheet.append_row(results)
    print("Entry logged successfully!")