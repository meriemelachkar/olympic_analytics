import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Chargement et prétraitement des données
df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')
df = preprocessor.preprocess(df, region_df)

# Configuration de la barre latérale
st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://th.bing.com/th/id/R.3b1e44147d6272f41f0be9a7e60e0835?rik=IJlpZ9Fl9j99dg&riu=http%3a%2f%2fclipart-library.com%2fnew_gallery%2f171-1711358_olympic-symbol-png-winter-olympics-logo.png&ehk=UbjSXoHUtC9NuVIKfkNXiorRcGRHQzJZ0OKC6BqbhkU%3d&risl=&pid=ImgRaw&r=0')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 
     'Athlete wise Analysis', 'Predictions')
)

# Section 1: Tableau des médailles
if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years, country = helper.country_year_list(df)
    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)
    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)

    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Tally")
    elif selected_year != 'Overall' and selected_country == 'Overall':
        st.title(f"Medal Tally in {selected_year} Olympics")
    elif selected_year == 'Overall' and selected_country != 'Overall':
        st.title(f"{selected_country} overall performance")
    else:
        st.title(f"{selected_country} performance in {selected_year} Olympics")
    st.table(medal_tally)

# Section 2: Analyse globale
elif user_menu == 'Overall Analysis':
    editions = df['Year'].nunique() - 1
    cities = df['City'].nunique()
    sports = df['Sport'].nunique()
    events = df['Event'].nunique()
    athletes = df['Name'].nunique()
    nations = df['region'].nunique()

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events over the years")
    st.plotly_chart(fig)

    athlete_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    st.title("No. of Events over time (Every Sport)")
    fig, ax = plt.subplots(figsize=(20, 20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype(int),
                     annot=True)
    st.pyplot(fig)

    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    st.table(x)

# Section 3: Analyse par pays
elif user_menu == 'Country-wise Analysis':
    st.sidebar.title('Country-wise Analysis')
    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()
    selected_country = st.sidebar.selectbox('Select a Country', country_list)

    country_df = helper.yearwise_medal_tally(df, selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(f"{selected_country} Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(f"{selected_country} excels in the following sports")
    pt = helper.country_event_heatmap(df, selected_country)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = sns.heatmap(pt, annot=True)
    st.pyplot(fig)

    st.title(f"Top 10 athletes of {selected_country}")
    top10_df = helper.most_successful_countrywise(df, selected_country)
    st.table(top10_df)

# Section 4: Analyse des athlètes
elif user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])
    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    fig = ff.create_distplot([x1, x2, x3, x4],
                             ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    st.title("Height Vs Weight")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')
    selected_sport = st.selectbox('Select a Sport', sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)
    fig, ax = plt.subplots()
    ax = sns.scatterplot(x=temp_df['Weight'], y=temp_df['Height'],
                         hue=temp_df['Medal'], style=temp_df['Sex'], s=60)
    st.pyplot(fig)

    st.title("Men Vs Women Participation Over the Years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=["Male", "Female"])
    st.plotly_chart(fig)

# Section 5: Prévisions
elif user_menu == 'Predictions':
    st.title("Prévision des médailles olympiques")
    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()
    selected_country = st.selectbox('Sélectionnez un pays', country_list)

    predictions = helper.predict_medals(df, selected_country)
    metrics = helper.get_prediction_metrics(df, selected_country)

    st.subheader("Statistiques historiques")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Moyenne de médailles", metrics['average_medals'])
    with col2:
        st.metric("Maximum de médailles", metrics['max_medals'])
    with col3:
        st.metric("Minimum de médailles", metrics['min_medals'])
    with col4:
        st.metric("Total des médailles", metrics['total_medals'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predictions['historical_years'],
        y=predictions['historical_medals'],
        name='Historique',
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=predictions['future_years'],
        y=predictions['predictions'],
        name='Prévisions',
        mode='markers',
        marker=dict(size=10, symbol='star')
    ))
    fig.update_layout(
        title=f"Prévision des médailles pour {selected_country}",
        xaxis_title="Année",
        yaxis_title="Nombre de médailles",
        hovermode='x'
    )
    st.plotly_chart(fig)

    st.subheader("Prévisions détaillées")
    predictions_df = pd.DataFrame({
        'Année': predictions['future_years'],
        'Médailles prévues': predictions['predictions']
    })
    st.table(predictions_df)

    st.info("""
    Note sur les prévisions :
    - Ces prévisions sont basées sur une régression linéaire simple.
    - D'autres facteurs peuvent influencer les résultats réels.
    """)
