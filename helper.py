import numpy as np


def fetch_medal_tally(df, year, country):
    medal_df = df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'])
    flag = 0
    if year == 'Overall' and country == 'Overall':
        temp_df = medal_df
    if year == 'Overall' and country != 'Overall':
        flag = 1
        temp_df = medal_df[medal_df['region'] == country]
    if year != 'Overall' and country == 'Overall':
        temp_df = medal_df[medal_df['Year'] == int(year)]
    if year != 'Overall' and country != 'Overall':
        temp_df = medal_df[(medal_df['Year'] == year) & (medal_df['region'] == country)]

    if flag == 1:
        x = temp_df.groupby('Year').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Year').reset_index()
    else:
        x = temp_df.groupby('region').sum()[['Gold', 'Silver', 'Bronze']].sort_values('Gold',
                                                                                      ascending=False).reset_index()

    x['total'] = x['Gold'] + x['Silver'] + x['Bronze']

    x['Gold'] = x['Gold'].astype('int')
    x['Silver'] = x['Silver'].astype('int')
    x['Bronze'] = x['Bronze'].astype('int')
    x['total'] = x['total'].astype('int')

    return x


def country_year_list(df):
    years = df['Year'].unique().tolist()
    years.sort()
    years.insert(0, 'Overall')

    country = np.unique(df['region'].dropna().values).tolist()
    country.sort()
    country.insert(0, 'Overall')

    return years,country

def data_over_time(df, col):
    # First get count of unique entries per year
    nations_over_time = df.drop_duplicates(['Year', col]).groupby('Year')[col].count().reset_index()
    # Rename Year to Edition for clarity
    nations_over_time.rename(columns={'Year': 'Edition'}, inplace=True)
    # Sort by year/edition
    nations_over_time = nations_over_time.sort_values('Edition')
    
    return nations_over_time

def most_successful(df, sport):
    # 1. First drop NA medals
    temp_df = df.dropna(subset=['Medal'])

    # 2. Filter by sport if needed
    if sport != 'Overall':
        temp_df = temp_df[temp_df['Sport'] == sport]

    # 3. Get medal counts
    medal_counts = temp_df['Name'].value_counts().to_frame().reset_index()
    
    # 4. Rename columns explicitly
    medal_counts.columns = ['Name', 'Medals']
    
    # 5. Get top 15
    top_15 = medal_counts.head(15)
    
    # 6. Get athlete details
    athlete_df = df[['Name', 'Sport', 'region']].drop_duplicates(subset=['Name'])
    
    # 7. Merge the data
    result = top_15.merge(
        athlete_df,
        on='Name',
        how='left'
    )
    
    # 8. Final cleanup and column ordering
    final = result[['Name', 'Medals', 'Sport', 'region']].drop_duplicates()
    
    return final

def yearwise_medal_tally(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]
    final_df = new_df.groupby('Year').count()['Medal'].reset_index()

    return final_df

def country_event_heatmap(df,country):
    temp_df = df.dropna(subset=['Medal'])
    temp_df.drop_duplicates(subset=['Team', 'NOC', 'Games', 'Year', 'City', 'Sport', 'Event', 'Medal'], inplace=True)

    new_df = temp_df[temp_df['region'] == country]

    pt = new_df.pivot_table(index='Sport', columns='Year', values='Medal', aggfunc='count').fillna(0)
    return pt


def most_successful_countrywise(df, country):
    # Filter for medals and country
    temp_df = df.dropna(subset=['Medal'])
    temp_df = temp_df[temp_df['region'] == country]
    
    # Get medal counts
    medal_counts = temp_df['Name'].value_counts().to_frame().reset_index()
    medal_counts.columns = ['Name', 'Medals']
    
    # Get top 10
    top_10 = medal_counts.head(10)
    
    # Get athlete details
    athlete_df = df[['Name', 'Sport']].drop_duplicates(subset=['Name'])
    
    # Merge and clean up
    result = top_10.merge(
        athlete_df,
        on='Name',
        how='left'
    )
    
    final = result[['Name', 'Medals', 'Sport']].drop_duplicates()
    
    return final

def weight_v_height(df,sport):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    athlete_df['Medal'].fillna('No Medal', inplace=True)
    if sport != 'Overall':
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        return temp_df
    else:
        return athlete_df

def men_vs_women(df):
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    men = athlete_df[athlete_df['Sex'] == 'M'].groupby('Year').count()['Name'].reset_index()
    women = athlete_df[athlete_df['Sex'] == 'F'].groupby('Year').count()['Name'].reset_index()

    final = men.merge(women, on='Year', how='left')
    final.rename(columns={'Name_x': 'Male', 'Name_y': 'Female'}, inplace=True)

    final.fillna(0, inplace=True)

    return final