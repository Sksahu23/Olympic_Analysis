import pandas as pd
import streamlit as st
import helper
import preprocessor
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')
df = preprocessor.preprocess(df, region_df)


def create_over_time_chart(df, col, title):
    flag = 0
    if col == 'Name':
        flag = 1
    data = helper.data_over_time(df, col)
    data.rename(columns={'Year': 'edition', 'count': col}, inplace=True)

    fig = px.line(data, x='edition', y=col, markers=True, 
                  line_shape='linear', 
                  color_discrete_sequence=['#636EFA'],
                  template='plotly_dark')

    if flag == 1:
        yaxis_title = 'Total Athletes'
    else:
        yaxis_title = "Total " + col + 's'

    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title="Year",
        yaxis_title=yaxis_title,
        autosize=False,
        width=800,
        height=500,
        margin=dict(l=50, r=50, b=50, t=50, pad=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_traces(marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey')),
                      line=dict(width=4))

    st.plotly_chart(fig)


st.sidebar.title("Olympics Analysis")
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT6gd2HIiptwQ4jOKtC9Gii1AVsrUg9JyQZBg&s")
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 'Athlete wise Analysis', 'Athlete comparison')
)
# 1) MEDAL TALLY : -------------------------------------

if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years, country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", country)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)
    medal_tally.index = medal_tally.index + 1
    
    if selected_year == 'Overall' and selected_country == 'Overall':
        st.title("Overall Medal Tally")
    if selected_year != 'Overall' and selected_country == 'Overall':
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " overall performance")
    if selected_year != 'Overall' and selected_country != 'Overall':
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")
    st.table(medal_tally)


# 2) OVERALL ANALYSIS : --------------------

if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

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

# 2.1
    create_over_time_chart(df, 'region', 'Participating nations over the Years')
# 2.2 
    create_over_time_chart(df, 'Event', 'Events over the Years')
# 2.3
    create_over_time_chart(df, 'Name', 'Athletes over the Years')

# 2.4
    st.title('Number of Events over the Year')
    fig, ax = plt.subplots(figsize=(15, 20))
    sk = df.drop_duplicates(subset=['Year', 'Sport', 'Event'])
    events_per_year = sk.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype(
        'int')

    ax = sns.heatmap(events_per_year, annot=True, cmap='flare', linewidths=.5, linecolor='black')

    ax.set_title('Heatmap of Number of Events Over the Years by Sport', fontsize=20, pad=20, color='white')
    ax.set_xlabel('Year', fontsize=16, color='white')
    ax.set_ylabel('Sport', fontsize=16, color='white')

    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    ax.tick_params(colors='white', which='both')
    plt.xticks(rotation=45, color='white')
    plt.yticks(rotation=0, color='white')

    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(colorbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    st.pyplot(fig)

# 2.5 
    st.title("Top 10 Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    selected_sport = st.selectbox('Select a Sport', sport_list)
    x = helper.most_successful(df, selected_sport)
    st.table(x)


# 3) COUNTRY WISE ANALYSIS : --------------

if user_menu == 'Country-wise Analysis':
    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()
    selected_country = st.sidebar.selectbox('Select a Country', country_list)

# 3.1
    country_df = helper.year_wise_medal_tally(df, selected_country)
    fig = px.line(country_df, x='Year', y='Medal',markers=True)
    fig.update_traces(marker=dict(size=10, symbol='circle', line=dict(width=2, color='DarkSlateGrey')))
    st.title(selected_country + " Medal Tally over the Years")
    st.plotly_chart(fig)

# 3.2
    st.title(f"{selected_country} excels in the following sports")

    pt = helper.country_event_heatmap(df, selected_country)
    fig, ax = plt.subplots(figsize=(17, 20))
    fig.patch.set_alpha(0.0)
    sns.set(font_scale=1.3) 
    ax = sns.heatmap(pt, annot=True, cmap='flare', linewidths=.5, linecolor='black', cbar_kws={'label': 'Number of Medals'})

    ax.set_title(f"Heatmap of {selected_country}'s Performance in Various Sports", fontsize=20, pad=20, color='white')
    ax.set_xlabel('Year', fontsize=16, color='white')
    ax.set_ylabel('Sports', fontsize=16, color='white')
    ax.set_facecolor('none')
    ax.tick_params(colors='white', which='both') 
    plt.xticks(rotation=0, color='white')
    plt.yticks(rotation=0, color='white')

    colorbar = ax.collections[0].colorbar
    colorbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(colorbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    st.pyplot(fig)

# 3.3
    st.title("Top 10 Athletes of " + selected_country)
    top10_df = helper.most_successful_country_wise(df, selected_country)
    st.table(top10_df)

# 4) ATHLETE WISE ANALYSIS : --------------------------------

if user_menu == 'Athlete wise Analysis':
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    x1 = athlete_df['Age'].dropna()
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

# 4.1
    fig = ff.create_distplot([x1, x2, x3, x4], ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=800, height=600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []
    famous_sports = ['Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
                     'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
                     'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
                     'Water Polo', 'Hockey', 'Rowing', 'Fencing',
                     'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
                     'Tennis', 'Golf', 'Softball', 'Archery',
                     'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
                     'Rhythmic Gymnastics', 'Rugby Sevens',
                     'Beach Volleyball', 'Triathlon', 'Rugby', 'Polo', 'Ice Hockey']
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

# 4.2
    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)
    fig.update_layout(autosize=False, width=800, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)

# 4.3
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')
 
    selected_sport = st.selectbox('Select a Sport', sport_list)
    st.title(f"Scatter Plot of Weight vs. Height for {selected_sport}")
    temp_df = helper.weight_v_height(df, selected_sport)

    fig, ax = plt.subplots()
    sns.set(style='darkgrid')
    ax = sns.scatterplot(data=temp_df, x='Weight', y='Height', hue='Medal', style='Sex', s=40)
    fig.patch.set_alpha(0.0)

    ax.set_xlabel('Year', fontsize=12, color='white')
    ax.set_ylabel('Sports', fontsize=12, color='white')
    ax.set_facecolor('none')

    ax.tick_params(colors='white', which='both') 
    plt.xticks(rotation=0, color='white', fontsize=10)
    plt.yticks(rotation=0, color='white', fontsize=10)

    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

# 4.4
    final = helper.men_vs_women(df)
    fig = px.line(
        final, 
        x="Year", 
        y=["Male", "Female"], 
        color_discrete_map={'Male': 'blue', 'Female': 'red'},
        markers=True
    )
    fig.update_layout(
        title={
            'text': "Men Vs Women Participation Over the Years",
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24} 
        },
        xaxis_title="Year",
        yaxis_title="Number of Participants",
        legend_title="Gender",
        autosize=False,
        width=800,
        height=600,
        template="plotly_white"
    )
    fig.update_traces(
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=8)
    )
    st.plotly_chart(fig)


# 5) ATHLETE COMPARISON : ----------------

if user_menu == 'Athlete comparison':

    def get_unique_athletes(df):
        athletes = df['Name'].unique().tolist()
        athletes.sort()
        return athletes

    def compare_athletes(df, athlete1, athlete2):

        athlete1_data = df[df['Name'] == athlete1]
        athlete2_data = df[df['Name'] == athlete2]

        athlete1_medals = athlete1_data['Medal'].value_counts().to_dict()
        athlete2_medals = athlete2_data['Medal'].value_counts().to_dict()

        athlete1_total_medals = athlete1_medals.get('Gold', 0) + athlete1_medals.get('Silver', 0) + athlete1_medals.get('Bronze', 0)
        athlete2_total_medals = athlete2_medals.get('Gold', 0) + athlete2_medals.get('Silver', 0) + athlete2_medals.get('Bronze', 0)

        athlete1_age = round(athlete1_data['Age'].mean(),1)
        athlete2_age = round(athlete2_data['Age'].mean(),1)

        athlete1_height = round(athlete1_data['Height'].mean(),1)
        athlete2_height = round(athlete2_data['Height'].mean(),1)

        athlete1_weight = round(athlete1_data['Weight'].mean(),1)
        athlete2_weight = round(athlete2_data['Weight'].mean(),1)

        athlete1_region = athlete1_data['region'].unique().tolist()
        athlete2_region = athlete2_data['region'].unique().tolist()

# 5.1
        comparison = pd.DataFrame({
            'Metric': ['Total Medals', 'Gold', 'Silver', 'Bronze', 'Average Age', 'Average Height', 'Average Weight', 'Country'],
            athlete1: [
                athlete1_total_medals,
                athlete1_medals.get('Gold', 0),
                athlete1_medals.get('Silver', 0),
                athlete1_medals.get('Bronze', 0),
                athlete1_age,
                f"{athlete1_height} cm",
                f"{athlete1_weight} Kg",
                athlete1_region[-1] if athlete1_region else 'N/A'
            ],
            athlete2: [
                athlete2_total_medals,
                athlete2_medals.get('Gold', 0),
                athlete2_medals.get('Silver', 0),
                athlete2_medals.get('Bronze', 0),
                athlete2_age,
                f"{athlete2_height} cm",
                f"{athlete2_weight} Kg",
                athlete2_region[-1] if athlete2_region else 'N/A'
            ]
        })
        return comparison


    st.title("Athlete Comparison")

    athletes = get_unique_athletes(df)
    athlete1 = st.selectbox("Select Athlete 1", athletes)
    athlete2 = st.selectbox("Select Athlete 2", athletes, index=1)

    if athlete1 and athlete2:
        comparison_data = compare_athletes(df, athlete1, athlete2)
        st.dataframe(comparison_data)

# 5.2
    def calculate_performance_metrics(df, athlete_name):
        athlete_data = df[df['Name'] == athlete_name]
        medals = athlete_data.groupby('Event')['Medal'].value_counts().unstack(fill_value=0)

        for medal in ['Gold', 'Silver', 'Bronze']:
            if medal not in medals.columns:
                medals[medal] = 0

        medals['Total_Medals'] = medals['Gold'] + medals['Silver'] + medals['Bronze']
        
        event_counts = athlete_data['Event'].value_counts().reset_index()
        event_counts.columns = ['Event', 'Participations']

        performance_data = medals.merge(event_counts, on='Event', how='outer').fillna(0)
        performance_data['Medals_Per_Participation'] = performance_data['Total_Medals'] / performance_data['Participations']

        return performance_data

    selected_athlete = st.selectbox("Select an Athlete", athletes)

    if selected_athlete:

        st.subheader("Performance Analysis")
        performance_data = calculate_performance_metrics(df, selected_athlete)
        st.dataframe(performance_data)

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=performance_data['Event'],
            y=performance_data['Gold'],
            name='Gold Medals',
            marker_color='gold'
        ))
        fig.add_trace(go.Bar(
            x=performance_data['Event'],
            y=performance_data['Participations'],
            name='Participations',
            marker_color='grey'
        ))   
        fig.update_layout(
            title={
                'text': f'Performance Analysis for {selected_athlete}',
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24}
            },
            xaxis_title='Events',
            yaxis_title='Count',
            barmode='group',
            width = 850,
            height = 700
        )
        st.plotly_chart(fig)
