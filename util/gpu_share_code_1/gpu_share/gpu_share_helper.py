import os.path
import datetime
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from simple_term_menu import TerminalMenu

class GpuShareHelper:
    def __init__(self):
        self.name = ""
        self.machine = None
        self.machine_rep = []

        for machine in self.machine_props:
            self.machine_rep.append(f"{machine['machine_name']} - {machine['gpu_count']} {machine['gpu_memory']}GB {machine['gpu_type']} GPUs")

        self.gpus = []
        self.usage_duration = 0
        self.event_id = ""

        self.sheets_service = None
        self.calendar_service = None

    @property
    def scopes(self):
        return ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/calendar']

    @property
    def sheet_id(self):
        return "1vGx-clptgrbwZzbmbyac1uR8YEivsE_pcsjwAqsNKEg"

    @property
    def calendar_id(self):
        return "33ce7cdfmrnjlua5j2v2i2bnc8@group.calendar.google.com"

    @property
    def machine_props(self):
        return [
            {
                "machine_name": "ece-dnc204168D",
                "machine_tab_id": "0",
                "gpu_count": 2,
                "gpu_memory": 16,
                "gpu_type": "RTX 5000",
            },
            {
                "machine_name": "ece-dnc244503D",
                "machine_tab_id": "2057975239",
                "gpu_count": 1,
                "gpu_memory": 48,
                "gpu_type": "RTX A6000",
            },
            {
                "machine_name": "ece-dnc244504D",
                "machine_tab_id": "340510035",
                "gpu_count": 1,
                "gpu_memory": 48,
                "gpu_type": "RTX A6000",
            },
            {
                "machine_name": "ece-dnc244505D",
                "machine_tab_id": "28240418",
                "gpu_count": 2,
                "gpu_memory": 16,
                "gpu_type": "RTX 5000",
            },
            {
                "machine_name": "ece-d01181714S",
                "machine_tab_id": "501470054",
                "gpu_count": 4,
                "gpu_memory": 32,
                "gpu_type": "V100",
            },
            {
                "machine_name": "ece-c01187299s",
                "machine_tab_id": "1589186071",
                "gpu_count": 4,
                "gpu_memory": 80,
                "gpu_type": "A100",
            },
        ]

    def setup(self):
        """
        This function adds the user's name to the appropriate sheet and adds a chunk to the calendar.
        """
        self._authenticate()

        if os.path.exists('current_usage.json'):
            print("LOADING GPU SHARE DATA FROM FILE")
            with open('current_usage.json') as json_file:
                json_string = json.load(json_file)
                data = json.loads(json_string)
                self.name = data['name']
                self.event_id = data['eventId']
                self.machine = data['machine']
                self.gpus = data['gpus']

        else:
            print("FRESH GPU SHARE SETUP")
            self._get_info()
            self._update_sheets(is_init=True)
            self._update_calendar()

            json_string = json.dumps({
                'name': self.name,
                'eventId': self.event_id,
                'machine': self.machine,
                'gpus': self.gpus
            })

            with open('current_usage.json', 'w') as outfile:
                json.dump(json_string, outfile)

    def cleanup(self):
        """
        This function removes the user's name from the appropriate sheet and the time chunk from the calendar.
        """
        self._update_sheets(is_init=False)
        self._update_calendar()
        os.remove('current_usage.json')

    def _get_info(self):
        """
        This function prompts the user to enter all pertinent data to update the GPU sharing documentation.
        """

        # Get the user's name
        self.name = input("Please enter your name: ")
        print("")

        # Get the user's machine
        terminal_menu = TerminalMenu(self.machine_rep)
        choice_index = terminal_menu.show()
        self.machine = self.machine_props[choice_index]
        print(f"You chose: {self.machine['machine_name']}\n")

        # Get the user's gpus
        gpus_not_entered = True
        while gpus_not_entered:
            temp_gpus_input = input("Please enter the gpus you will be using as a comma separated list (i.e. 0,1 or 0,1,2 or 0,3): ")
            try:
                self.gpus = [int(i) for i in temp_gpus_input.replace(" ", "").split(',')]
                if max(self.gpus) < self.machine['gpu_count'] and min(self.gpus) >= 0:
                    gpus_not_entered = False
                else:
                    print("You tried to select a GPU that does not exist on this machine. Please try again.")
            except Exception as e:
                print("Something went wrong, please try again.")

        self.usage_duration = int(input("Estimate the number of days you will be using the resources: "))

    def _authenticate(self):
        """
        This function authenticates the user in order to access the Google Sheets and Calendar APIs
        """
        creds = None

        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', self.scopes)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    '../../util/gpu_share_code/gpu_share/credentials.json', self.scopes)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        try:
            self.sheets_service = build('sheets', 'v4', credentials=creds)
            self.calendar_service = build('calendar', 'v3', credentials=creds)
        except HttpError as err:
            print(err)

    def _update_sheets(self, is_init):
        """
        This function updates the group sheet with the usage of the selected machine and GPUs.
        Args
            is_init: (bool) whether or not the function should add the entries, or clear them
        """
        
        values = []
        cells_to_change = f"{self.machine['machine_name']}!A2:A{1 + self.machine['gpu_count']}"
        cells_to_read = f"{self.machine['machine_name']}!A2:B{1 + self.machine['gpu_count']}"
        
        #Get the current values in the spreadsheet
        result = self.sheets_service.spreadsheets().values().get(spreadsheetId=self.sheet_id, 
                                                                 range=cells_to_read).execute()
        current_values = result.get('values',[])
        #print(current_values)

        for i in range(self.machine["gpu_count"]):
            if i in self.gpus:
                values.append([self.name if is_init else ""])
            else:
                values.append([current_values[i][0]])


        HTTP_REQUEST_BODY = {
            'values': values,
        }

        try:
            self.sheets_service.spreadsheets().values().update(
                spreadsheetId=self.sheet_id, range=cells_to_change,
                valueInputOption='RAW', body=HTTP_REQUEST_BODY).execute()
            print("Successfully updated the Google Sheet.")
        except Exception as e:
            print(e)
            print("There was an error updating the sheet - please do so manually.")

    def _update_calendar(self):
        """
        This function updates the group calendar with the usage of the selected machine and GPUs.
        """
        if self.event_id != "":
            try:
                self.calendar_service.events().delete(calendarId=self.calendar_id, eventId=self.event_id).execute()
                print("Successfully removed your chunk from the calendar.")
            except Exception as e:
                print(e)
                print("Was unable to remove the event to the calendar - please do so mannually.")

            return

        start = datetime.datetime.utcnow()
        end = start + datetime.timedelta(days=self.usage_duration-1)

        event = {
            'summary': f'{self.name} using {len(self.gpus)} GPUs on {self.machine["machine_name"]}',
            'start': {
                'dateTime': start.isoformat(),
                'timeZone': 'America/New_York',
            },
            'end': {
                'dateTime': end.isoformat(),
                'timeZone': 'America/New_York',
            },
        }

        try:
            respone = self.calendar_service.events().insert(calendarId=self.calendar_id, body=event).execute()
            self.event_id = respone['id']
            print("Successfully updated the calendar with your GPU usage.")
        except Exception as e:
            print(e)
            print("Was unable to add the event to the calendar - please do so mannually.")
