import pandas as pd
import os
import numpy as np

# --- FUNZIONI DI CARICAMENTO ---

def crea_dataframe_partite(data_path: str = "data") -> pd.DataFrame:
    all_dataframes = []
    if not os.path.isdir(data_path):
        print(f"Errore: Cartella '{data_path}' non trovata.")
        return pd.DataFrame()
    
    for file in os.listdir(data_path):
        if "atp_matches" in file and file.endswith(".csv"):
            full_path = os.path.join(data_path, file)
            
            df_partite = pd.read_csv(full_path)
            all_dataframes.append(df_partite)
        
    return pd.concat(all_dataframes, ignore_index=True)

def crea_dataframe_giocatori(data_path: str = "data") -> pd.DataFrame:
    if not os.path.isdir(data_path):
        return pd.DataFrame()
    
    for file in os.listdir(data_path):
        if "players" in file and file.endswith(".csv"):
            full_path = os.path.join(data_path, file)
            return pd.read_csv(full_path)
    return pd.DataFrame()

def crea_dataframe_ranking(data_path: str = "data") -> pd.DataFrame:
    all_dataframes = []
    if not os.path.isdir(data_path):
        return pd.DataFrame()
        
    for file in os.listdir(data_path):
        if "rankings" in file and file.endswith(".csv"):
            full_path = os.path.join(data_path, file)
            
            df_ranking = pd.read_csv(full_path)
            all_dataframes.append(df_ranking)
    return pd.concat(all_dataframes, ignore_index=True)

# --- FUNZIONI DI PULIZIA ---

def pulisci_dataframe_partite(df: pd.DataFrame) -> pd.DataFrame:
    df["tourney_date"] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors="coerce")
    return df

def pulisci_dataframe_giocatori(df: pd.DataFrame) -> pd.DataFrame:
    df["dob"] = pd.to_datetime(df["dob"], format='%Y%m%d', errors="coerce")
    
    df.loc[~df['height'].between(150, 220), 'height'] = np.nan
    return df

def pulisci_dataframe_ranking(df: pd.DataFrame) -> pd.DataFrame:
    df["ranking_date"] = pd.to_datetime(df["ranking_date"], format="%Y%m%d", errors="coerce")
    return df



def pulisci_eta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce le colonne 'winner_age' e 'loser_age' già presenti nel dataset.
    Imposta a NaN i valori sotto i 14 anni o sopra i 49 anni.
    """
    colonne_eta = ['winner_age', 'loser_age']
    
    for col in colonne_eta:
        if col in df.columns:
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            

            df.loc[~df[col].between(14, 49), col] = np.nan
            
    return df

def aggiungi_ranking(df_partite: pd.DataFrame, df_ranking: pd.DataFrame) -> pd.DataFrame:
    df_partite = df_partite.sort_values("tourney_date")
    df_ranking = df_ranking.sort_values("ranking_date")

    # Merge per vincitore
    df_partite = pd.merge_asof(df_partite, df_ranking, left_on="tourney_date", right_on="ranking_date", left_by="winner_id", right_by="player", direction="backward").rename(columns={"rank": "winner_rank_new", "points": "winner_points_new"})
    
    df_partite = df_partite.drop(columns=["player", "ranking_date"], errors="ignore")

    # Merge per perdente
    df_partite = pd.merge_asof(df_partite, df_ranking, left_on="tourney_date", right_on="ranking_date", left_by="loser_id", right_by="player", direction="backward").rename(columns={"rank": "loser_rank_new", "points": "loser_points_new"})
    
    df_partite = df_partite.drop(columns=["player", "ranking_date"], errors="ignore")
    return df_partite

# --- FUNZIONI DI ANALISI ---

def analizza_head_to_head(df_partite: pd.DataFrame, giocatore1: str, giocatore2: str) -> None:
    # Filtro case-insensitive
    g1_matches = df_partite[
        (df_partite['winner_name'].str.contains(giocatore1, case=False, na=False)) | 
        (df_partite['loser_name'].str.contains(giocatore1, case=False, na=False))
    ]
    
    scontri = g1_matches[
        (g1_matches['winner_name'].str.contains(giocatore2, case=False, na=False)) | 
        (g1_matches['loser_name'].str.contains(giocatore2, case=False, na=False))
    ]
    
    if scontri.empty:
        print(f"\nNessuno scontro trovato tra '{giocatore1}' e '{giocatore2}'.")
        return

    vittorie_g1 = scontri[scontri['winner_name'].str.contains(giocatore1, case=False, na=False)].shape[0]
    vittorie_g2 = scontri[scontri['winner_name'].str.contains(giocatore2, case=False, na=False)].shape[0]
    
    print(f"\nStatistiche Scontri Diretti: {giocatore1} vs {giocatore2}")
    print(f"Totale Partite: {len(scontri)}")
    print(f"Vittorie {giocatore1}: {vittorie_g1}")
    print(f"Vittorie {giocatore2}: {vittorie_g2}")
    print("-" * 30)
    print("Ultimi 5 incontri:")
    
    ultimi = scontri.sort_values('tourney_date', ascending=False).head(5)
    for _, row in ultimi.iterrows():
        anno = row['tourney_date'].year if pd.notnull(row['tourney_date']) else "N/A"
        print(f"{anno} - {row['tourney_name']}: Vince {row['winner_name']} ({row['score']})")


def calcola_statistiche_carriera(df_partite: pd.DataFrame) -> pd.DataFrame:
    # 1. Statistiche base
    stats_w = df_partite.groupby('winner_name').agg({
        'winner_id': 'count',
        'w_ace': 'sum'
    }).rename(columns={'winner_id': 'vittorie', 'w_ace': 'ace_vittorie'})
    
    stats_l = df_partite.groupby('loser_name').agg({
        'loser_id': 'count',
        'l_ace': 'sum'
    }).rename(columns={'loser_id': 'sconfitte', 'l_ace': 'ace_sconfitte'})
    
    # 2. Calcolo Titoli (Finali vinte)
    finali_vinte = df_partite[df_partite['round'] == 'F']['winner_name'].value_counts()
    finali_vinte.name = 'titoli'
    
    # 3. NUOVO: Calcolo Vittorie contro Top 10
    # Filtriamo le partite dove il perdente aveva rank <= 10
    vittorie_top10 = df_partite[df_partite['loser_rank_new'] <= 10]['winner_name'].value_counts()
    vittorie_top10.name = 'top10_wins'
    
    # 4. Merge di tutto
    stats_totali = stats_w.join(stats_l, how='outer').fillna(0)
    stats_totali = stats_totali.join(finali_vinte, how='left').fillna(0)
    stats_totali = stats_totali.join(vittorie_top10, how='left').fillna(0) # Aggiungiamo le top10 wins
    
    # 5. Calcoli finali
    stats_totali['totale_partite'] = stats_totali['vittorie'] + stats_totali['sconfitte']
    stats_totali['percentuale_vittoria'] = (stats_totali['vittorie'] / stats_totali['totale_partite'] * 100).round(2)
    stats_totali['totale_ace'] = stats_totali['ace_vittorie'] + stats_totali['ace_sconfitte']
    
    stats_totali = stats_totali.drop(columns=['ace_vittorie', 'ace_sconfitte'])
    stats_totali = stats_totali.sort_values('vittorie', ascending=False)
    
    return stats_totali

def visualizza_profilo_giocatore(df_stats: pd.DataFrame, df_players: pd.DataFrame, nome_cercato: str) -> None:
    risultati = df_stats[df_stats.index.str.contains(nome_cercato, case=False, na=False)]
    
    if risultati.empty:
        print(f"Nessun giocatore trovato con il nome: {nome_cercato}")
        return
    
    # Creazione colonna nome completo per confronto
    if 'full_name' not in df_players.columns:
        df_players['full_name'] = df_players['name_first'].astype(str) + " " + df_players['name_last'].astype(str)

    print(f"\nTrovati {len(risultati)} risultati:")
    
    for nome_completo, dati in risultati.iterrows():
        print(f"\n--- {nome_completo.upper()} ---")
        
        # Dati Anagrafici
        player_info = df_players[df_players['full_name'] == nome_completo]
        if not player_info.empty:
            info = player_info.iloc[0]
            
            altezza = f"{int(info['height'])} cm" if pd.notnull(info['height']) else "N/D"
            mano = info['hand'] if pd.notnull(info['hand']) else "N/D"
            nazionalita = info['ioc']
            
            eta_str = "N/D"
            if pd.notnull(info['dob']):
                eta = (pd.Timestamp.now() - info['dob']).days / 365.25
                eta_str = f"{int(eta)} anni"
            
            print(f"Eta:         {eta_str}")
            print(f"Nazionalita: {nazionalita}")
            print(f"Altezza:     {altezza}")
            print(f"Mano:        {mano}")
        
        # Statistiche
        print("-" * 20)
        print(f"Titoli ATP:  {int(dati['titoli'])}")
        print(f"Vittorie:    {int(dati['vittorie'])}")
        print(f"Sconfitte:   {int(dati['sconfitte'])}")
        print(f"Perc. Vitt:  {dati['percentuale_vittoria']}%")
        print(f"Ace Totali:  {int(dati['totale_ace'])}")

# --- MAIN BLOCK ---

if __name__ == "__main__":
    DATA_PATH = "data"

    df_matches = crea_dataframe_partite(DATA_PATH)
    df_players = crea_dataframe_giocatori(DATA_PATH)
    df_rankings = crea_dataframe_ranking(DATA_PATH)
    
    if df_matches.empty:
        print("Errore: Nessun dato partite trovato.")
        exit()

    # Pipeline pulizia
    df_matches = pulisci_dataframe_partite(df_matches)
    df_players = pulisci_dataframe_giocatori(df_players)
    df_rankings = pulisci_dataframe_ranking(df_rankings)
    
    df_matches = pulisci_eta(df_matches)
    df_matches = aggiungi_ranking(df_matches, df_rankings)
    df_stats_carriera = calcola_statistiche_carriera(df_matches)
    
    print(f"Caricamento completato. {len(df_matches)} partite elaborate.")

    # Menu
    while True:
        print("\n=== ATP DATA ANALYSIS MENU ===")
        print("1. Head to Head")
        print("2. Profilo Giocatore")
        print("3. Top 10 Vittorie")
        print("0. Esci")
        
        scelta = input("\nInserisci opzione: ").strip()
        
        if scelta == "1":
            p1 = input("Giocatore 1: ")
            p2 = input("Giocatore 2: ")
            analizza_head_to_head(df_matches, p1, p2)
            
        elif scelta == "2":
            p = input("Nome Giocatore: ")
            visualizza_profilo_giocatore(df_stats_carriera, df_players, p)
            
        elif scelta == "3":
            print("\nTop 10 Giocatori per Vittorie:")
            print(df_stats_carriera[['vittorie', 'sconfitte', 'percentuale_vittoria', 'titoli']].head(10))
            
        elif scelta == "0":
            break
        
        else:
            print("Opzione non valida.")