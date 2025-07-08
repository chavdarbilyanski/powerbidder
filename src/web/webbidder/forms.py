from django import forms

class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label='Upload CSV File',
        required=True,
        help_text='Upload a CSV file with columns: Date, Hour, Price (EUR)'
    )
    max_battery_capacity = forms.FloatField(
        label='Max Battery Capacity (kWh)',
        min_value=0.1,
        initial=3900.0,
        required=True,
        help_text='Enter the maximum battery capacity in kWh (e.g., 3900.0)'
    )
    charge_discharge_rate = forms.FloatField(
        label='Charge/Discharge Rate (kW per time step)',
        min_value=0.1,
        initial=1000.0,
        required=True,
        help_text='Enter the charge/discharge rate in kW per time step (e.g., 1000.0)'
    )