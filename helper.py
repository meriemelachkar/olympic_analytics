import numpy as np

def fetch_medal_tally(df, year, country):
    """
    Calcule le tableau des médailles en fonction de l'année et du pays sélectionnés.
    
    Args:
        df: DataFrame contenant les données olympiques
        year: Année sélectionnée ou 'Overall' pour toutes les années
        country: Pays sélectionné ou 'Overall' pour tous les pays
    
    Returns:
        DataFrame contenant le décompte des médailles
    """
    # Supprime les doublons pour éviter de compter plusieurs fois la même médaille
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    
    # Gestion des différents cas de filtrage
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == year) & (medal_df['region'] == country)]

    # Agrégation des résultats selon le flag
    if flag == 1:
        # Pour un pays spécifique, grouper par année
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        # Pour une année spécifique ou global, grouper par pays
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold', ascending=False).reset_index()

    # Calcul du total et conversion en entiers
    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']
    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')

    return x

def country_year_list(df):
    """
    Prépare les listes des années et pays pour les filtres de sélection.
    
    Args:
        df: DataFrame contenant les données olympiques
    
    Returns:
        Tuple contenant (liste des années, liste des pays)
    """
    # Création de la liste des années
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    # Création de la liste des pays
    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years, country

def data_over_time(df, col):
    """
    Analyse l'évolution d'une métrique au fil du temps.
    
    Args:
        df: DataFrame contenant les données olympiques
        col: Colonne à analyser
    
    Returns:
        DataFrame avec l'évolution de la métrique par année
    """
    nations_over_time = df.drop_duplicates(['Year', col]).groupby('Year')[col].count().reset_index()
    nations_over_time.rename(columns={'Year': 'Edition'}, inplace=True)
    nations_over_time = nations_over_time.sort_values('Edition')
    
    return nations_over_time

def most_successful(df, sport):
    """
    Identifie les athlètes les plus performants globalement ou par sport.
    
    Args:
        df: DataFrame contenant les données olympiques
        sport: Sport spécifique ou 'Overall' pour tous les sports
        
    Returns:
        DataFrame contenant les 15 athlètes les plus médaillés
    """
    temp_df = df.dropna(subset=['Medal'])
    
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    medal_counts = temp_df['Name'].value_counts().to_frame().reset_index()
    medal_counts.columns = ['Name', 'Medals']
    
    top_15 = medal_counts.head(15)
    athlete_df = df[['Name', 'Sport', 'region']].drop_duplicates(subset=['Name'])
    
    result = top_15.merge(athlete_df, on='Name', how='left')
    final = result[['Name', 'Medals', 'Sport', 'region']].drop_duplicates()
    
    return final

def yearwise_medal_tally(df, country):
    """
    Calcule le nombre total de médailles par année pour un pays donné.
    
    Args:
        df: DataFrame contenant les données olympiques
        country: Pays à analyser
        
    Returns:
        DataFrame avec le décompte des médailles par année
    """
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df, country):
    """
    Crée une table pivot pour visualiser les performances d'un pays par sport et par année.
    
    Args:
        df: DataFrame contenant les données olympiques
        country: Pays à analyser
        
    Returns:
        Table pivot des médailles par sport et année
    """
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    
    return pt

def most_successful_countrywise(df, country):
    """
    Identifie les athlètes les plus performants pour un pays spécifique.
    
    Args:
        df: DataFrame contenant les données olympiques
        country: Pays à analyser
        
    Returns:
        DataFrame contenant les 10 meilleurs athlètes du pays
    """
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]
    
    medal_counts = temp_df['Name'].value_counts().to_frame().reset_index()
    medal_counts.columns = ['Name', 'Medals']
    
    top_10 = medal_counts.head(10)
    athlete_df = df[['Name', 'Sport']].drop_duplicates(subset=['Name'])
    
    result = top_10.merge(athlete_df, on='Name', how='left')
    final = result[['Name', 'Medals', 'Sport']].drop_duplicates()
    
    return final

def weight_v_height(df, sport):
    """
    Prépare les données pour l'analyse poids/taille des athlètes.
    
    Args:
        df: DataFrame contenant les données olympiques
        sport: Sport spécifique ou 'Overall' pour tous les sports
        
    Returns:
        DataFrame filtré avec les données poids/taille
    """
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df

def men_vs_women(df):
    """
    Analyse la participation hommes/femmes aux Jeux Olympiques par année.
    
    Args:
        df: DataFrame contenant les données olympiques
        
    Returns:
        DataFrame avec le nombre d'athlètes hommes et femmes par année
    """
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)
    final.fillna(0, inplace=True)

    return final
from sklearn.linear_model import LinearRegression
def predict_medals(df, country):
    """
    Prédit le nombre de médailles futures pour un pays donné.
    
    Args:
        df: DataFrame contenant les données olympiques
        country: Pays pour lequel faire des prévisions
        
    Returns:
        Dictionnaire contenant les prédictions et les données historiques
    """
    # Filtrer les données pour le pays sélectionné
    country_df = df[df['region'] == country]
    
    # Compter le nombre de médailles par année
    medals_by_year = country_df.groupby('Year')['Medal'].count().reset_index()
    
    # Préparer les données pour le modèle
    X = medals_by_year['Year'].values.reshape(-1, 1)
    y = medals_by_year['Medal'].values
    
    # Créer et entraîner le modèle
    model = LinearRegression()
    model.fit(X, y)
    
    # Faire des prédictions pour les 3 prochaines olympiades
    last_year = X[-1][0]
    future_years = np.array([[last_year + 4], [last_year + 8], [last_year + 12]])
    predictions = model.predict(future_years)
    
    # Arrondir les prédictions à des nombres entiers
    predictions = np.maximum(0, predictions.round())
    
    return {
        'historical_years': X.flatten(),
        'historical_medals': y,
        'future_years': future_years.flatten(),
        'predictions': predictions,
        'model': model
    }

def get_prediction_metrics(df, country):
    """
    Calcule des métriques statistiques sur les performances passées.
    
    Args:
        df: DataFrame contenant les données olympiques
        country: Pays à analyser
        
    Returns:
        Dictionnaire contenant les métriques
    """
    country_df = df[df['region'] == country]
    medals_by_year = country_df.groupby('Year')['Medal'].count()
    
    return {
        'average_medals': round(medals_by_year.mean(), 2),
        'max_medals': int(medals_by_year.max()),
        'min_medals': int(medals_by_year.min()),
        'total_medals': int(medals_by_year.sum())
    }