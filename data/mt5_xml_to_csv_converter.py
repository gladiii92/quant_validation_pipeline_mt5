import pandas as pd

# Pfade
input_file = r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\data\raw\RangeBreakOut_USDJPY.xlsx"
output_file = r"G:\DAVID\Desktop\GitHub\quant_validation_pipeline_mt5\data\processed\RangeBreakOut_USDJPY_trades_merged.csv"

# Excel-Datei einlesen (alles erstmal, ohne Header)
df = pd.read_excel(input_file, sheet_name=0, header=None)

# ------------------ Orders-Block ------------------
# Zeile mit "Orders" finden
orders_title_idx = df[df.iloc[:, 0] == "Orders"].index[0]
orders_start_idx = orders_title_idx + 1  # +1, dann steht dort direkt der erste Order-Datensatz

# Zeile mit "Trades" finden, um das Ende des Orders-Blocks zu bestimmen
trades_title_idx = df[df.iloc[:, 0] == "Trades"].index[0]

orders_df_raw = df.iloc[orders_start_idx:trades_title_idx].copy()

# In deinem ursprünglichen Script wurden die Spalten so gemappt:
# headers = ['Eröffnungszeit','Auftrag','Symbol','Typ','Volumen','Leer1','Preis','S/L','T/P','Zeit','Leer2','Status','Kommentar']
# und dann auf relevante Spalten reduziert.
headers_full = ['Eröffnungszeit', 'Auftrag', 'Symbol', 'Typ', 'Volumen',
                'Leer1', 'Preis', 'S/L', 'T/P', 'Zeit', 'Leer2', 'Status', 'Kommentar']

# Es gibt 13 Spalten im Orders-Bereich, deshalb so zuweisen:
orders_df_raw = orders_df_raw.iloc[:, :len(headers_full)]
orders_df_raw.columns = headers_full

# Nur relevante Spalten behalten
orders_df = orders_df_raw[['Eröffnungszeit', 'Auftrag', 'Symbol', 'Typ',
                           'Volumen', 'Preis', 'S/L', 'T/P', 'Zeit', 'Status', 'Kommentar']]

# Leere Zeilen entfernen
orders_df = orders_df.dropna(how="all")

# ------------------ Trades-Block ------------------
trades_start_idx = trades_title_idx + 1  # Zeile nach "Trades"

trades_df_raw = df.iloc[trades_start_idx:].copy()

# Erste Zeile im Trades-Block ist die Headerzeile mit:
# Zeit, Trade, Symbol, Typ, Richtung, Volumen, Preis, Auftrag, Kommission, Swap, Gewinn, Kontostand, Kommentar
trades_headers = ["Zeit", "Trade", "Symbol", "Typ", "Richtung",
                  "Volumen", "Preis", "Auftrag", "Kommission",
                  "Swap", "Gewinn", "Kontostand", "Kommentar"]

# Header-Zeile holen und verwerfen
# Annahme: Direkt nach "Trades" kommt die Headerzeile
trades_header_row = trades_df_raw.iloc[0]
trades_df_raw = trades_df_raw.iloc[1:]  # Daten ab der nächsten Zeile

# Falls die Datei exakt so strukturiert ist, kannst du die Header hart setzen:
trades_df_raw = trades_df_raw.iloc[:, :len(trades_headers)]
trades_df_raw.columns = trades_headers

# Leere Zeilen entfernen
trades_df = trades_df_raw.dropna(how="all")

# ------------------ Mergen Orders + Trades ------------------
# Schlüssel: Trades.Trade (Trade-ID) <-> Orders.Auftrag (Order-ID)
merged = pd.merge(
    trades_df,
    orders_df,
    left_on="Trade",
    right_on="Auftrag",
    how="left",
    suffixes=("_trade", "_order")
)

# Zur Kontrolle einmal die Spalten drucken (zum Debuggen):
print("Spalten in merged:")
print(list(merged.columns))

# Beispielhafter Output: du kannst hier anpassen, welche Spalten du wirklich brauchst
# Wichtig: Namen exakt aus merged.columns übernehmen.
output_cols = [
    "Zeit",           # aus Trades-Block
    "Trade",          # Trade-ID
    "Symbol_trade",   # Symbol aus Trades
    "Typ_trade",      # Typ aus Trades
    "Richtung",
    "Volumen_trade",  # Volumen aus Trades
    "Preis_trade",    # Preis aus Trades
    "Auftrag",        # Order-ID (gleich wie Trade-Verknüpfung)
    "Gewinn",
    "Kontostand",
    "Eröffnungszeit", # aus Orders
    "Typ_order",      # Typ aus Orders
    "Preis_order",    # Preis aus Orders
    "S/L",
    "T/P",
    "Kommentar_trade",
    "Kommentar_order"
]

# Manche Spaltennamen existieren evtl. ohne Suffix, je nachdem, ob es Duplikate gab.
# Deshalb: Fallback-Liste dynamisch erzeugen.
available_cols = []
for col in output_cols:
    if col in merged.columns:
        available_cols.append(col)

# CSV speichern mit den verfügbaren Spalten
merged[available_cols].to_csv(output_file, index=False, sep=';', encoding='utf-8')

print(f"{len(merged)} Zeilen gespeichert in {output_file}")
