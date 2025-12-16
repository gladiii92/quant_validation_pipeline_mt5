import pandas as pd

# Pfade
input_file = r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\data\raw\RangeBreakOut_USDJPY.xlsx"
output_file = r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\data\processed\RangeBreakOut_USDJPY_trades.csv"

# Excel-Datei einlesen (alles erstmal)
df = pd.read_excel(input_file, sheet_name=0, header=None)

# "Orders"-Zeile finden
orders_start_idx = df[df.iloc[:, 0] == 'Orders'].index[0] + 1  # +1, um direkt die Header-Zeile zu nehmen
orders_df = df.iloc[orders_start_idx:]

# Header für die Trades definieren (wie sie in der Excel stehen)
headers = ['Eröffnungszeit','Auftrag','Symbol','Typ','Volumen','Leer1','Preis','S/L','T/P','Zeit','Leer2','Status','Kommentar']
orders_df.columns = headers

# Nur relevante Spalten behalten
orders_df = orders_df[['Eröffnungszeit','Auftrag','Symbol','Typ','Volumen','Preis','S/L','T/P','Zeit','Status','Kommentar']]

# CSV speichern
orders_df.to_csv(output_file, index=False, sep=';', encoding='utf-8')

print(f"{len(orders_df)} Trades gespeichert in {output_file}")
