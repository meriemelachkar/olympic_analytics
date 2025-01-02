import pandas as pd

def preprocess(df, region_df):
    """
    Prétraite les données olympiques en effectuant plusieurs opérations de nettoyage
    et de transformation.
    
    Args:
        df: DataFrame principal contenant les données des Jeux Olympiques
        region_df: DataFrame contenant les correspondances NOC-région
        
    Returns:
        DataFrame prétraité avec les données des Jeux d'été et les médailles encodées
        
    Opérations effectuées:
    1. Filtrage des Jeux d'été uniquement
    2. Fusion avec les données régionales
    3. Suppression des doublons
    4. Encodage one-hot des médailles
    """
    # Filtrage pour ne garder que les Jeux d'été
    # Cela exclut toutes les compétitions des Jeux d'hiver
    df = df[df['Season'] == 'Summer']
    
    # Fusion avec le DataFrame des régions
    # Permet d'ajouter les informations régionales basées sur le code NOC
    # 'left' garde toutes les lignes du DataFrame principal
    df = df.merge(region_df, on='NOC', how='left')
    
    # Suppression des entrées en double pour éviter les comptages multiples
    df.drop_duplicates(inplace=True)
    
    # Création de colonnes binaires pour chaque type de médaille (Gold, Silver, Bronze)
    # Utilise pd.get_dummies pour créer des colonnes 0/1 pour chaque type de médaille
    df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)
    
    return df