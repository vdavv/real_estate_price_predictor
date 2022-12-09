import streamlit as st
import pandas as pd
from scipy.stats import norm
import plotly.express as px
from numpy import *
from numerize.numerize import numerize


def parse_data():
    global data, data1
    data = pd.read_csv("data.csv")
    data1 = pd.read_csv("data.csv")


def clean_data(to_drop):
    global data, data1
    data = data.drop(columns=to_drop)
    data1 = data1.drop(columns=to_drop)


def round1(x=1):
    if round(x) % 2:
        return round(x)
    else:
        return 1 + round(x) if 1 + round(x) - x < x - round(x) + 1 else round(x) - 1


def round2(x=0):
    if round(x) % 2:
        return 1 + round(x) if 1 + round(x) - x < x - round(x) + 1 else round(x) - 1
    else:
        return round(x)


def plot_key_metrics():
    r = norm.rvs(size=1000)
    x_norm = linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
    fig = px.histogram(
        data["MEDV"],
        x="MEDV",
        nbins=100,
        histfunc="count",
        title="Visualisation",
        height=800, width=800
    )
    fig.update_yaxes(title="Number of houses")
    fig.update_xaxes(title="Price of house")
    fig.add_vline(
        x=mean_price,
        line_color="salmon",
        line_width=3,
        opacity=1,
        line_dash="dot",
    )
    fig.add_vline(
        x=median_price,
        line_color="lightgreen",
        line_width=3,
        opacity=1,
        line_dash="dot",
    )
    fig.add_vline(
        x=mean_price + std_price,
        line_color="red",
        line_width=2,
        opacity=1,
        line_dash="dot",
    )
    fig.add_vline(
        x=mean_price - std_price, line_color="red", line_width=2, opacity=1, line_dash="dot"
    )
    fig.add_annotation(text="mean", x=mean_price + 1.7, y=42, arrowhead=0, showarrow=False)
    fig.add_annotation(
        text="median", x=median_price - 2.1, y=42, arrowhead=0, showarrow=False
    )
    fig.add_shape(
        type="rect",
        y0=0,
        x0=mean_price - std_price,
        y1=46.7,
        x1=mean_price + std_price,
        opacity=0.1,
        fillcolor="red",
    )
    fig.add_annotation(
        text="mean+std",
        y="32.5",
        x=mean_price + std_price + 3,
        arrowhead=0,
        showarrow=False,
    )
    fig.add_annotation(
        text="mean-std",
        y="32.5",
        x=mean_price - std_price - 3,
        arrowhead=0,
        showarrow=False,
    )
    return fig


def plot_psc_nox_dis():
    fig = px.scatter(
        data_segmented,
        x="DIS",
        y="NOX",
        trendline="ols",
        color="TYPE",
        height=800,
        width=800,
        title="Nitric oxides concenration over Distance to the city centre",
        labels={
            "NOX": "Nitric oxides concentration around house",
            "DIS": "Distance from the city centre to the house",
        },
    )
    fig.update_layout(
        font_family="Rockwell",
        legend=dict(
            title=None, orientation="v", y=0.84, yanchor="bottom", x=0.855, xanchor="center"
        ),
    )
    return fig


def plot_psc_rm_age():
    fig = px.scatter(
        data_segmented,
        x="AGE",
        y="RM",
        trendline="ols",
        color="TYPE",
        height=800,
        width=800,
        title="Number of rooms in house over its Age",
        labels={
            "AGE": "Age of house",
            "RM": "Number of Rooms in house",
        },
    )
    fig.update_layout(
        font_family="Rockwell",
        legend=dict(
            title=None, orientation="v", y=0.87, yanchor="bottom", x=0.22, xanchor="center"
        ),
    )
    return fig


def plot_psc_age_dis():
    fig = px.scatter(
        data_segmented,
        x="DIS",
        y="AGE",
        trendline="ols",
        color="TYPE",
        height=800,
        width=800,
        title="Age of house over its Distance from the city centre",
        labels={
            "AGE": "Age of house",
            "DIS": "Distance from the house to the city centre",
        },
    )
    fig.update_layout(
        font_family="Rockwell",
        legend=dict(
            title=None, orientation="v", y=0.79, yanchor="bottom", x=0.8, xanchor="center"
        ),
    )
    fig.update_yaxes(range=[0, 105])
    return fig


def plot_psc_nox_age():
    fig = px.scatter(
        data_segmented,
        x="AGE",
        y="NOX",
        trendline="ols",
        color="TYPE",
        height=800,
        width=800,
        title="Nitric oxides concentration around the house over its Age",
        labels={
            "AGE": "Age of house",
            "NOX": "Nitric oxides concentration around the house",
        },
    )
    fig.update_layout(
        font_family="Rockwell",
        legend=dict(
            title=None, orientation="v", y=0.82, yanchor="bottom", x=0.3, xanchor="center"
        ),
    )
    return fig


def plot_psc_rm_dis():
    fig = px.scatter(
        data_segmented,
        x="DIS",
        y="RM",
        trendline="ols",
        color="TYPE",
        height=800,
        width=800,
        title="Number of rooms in the house over its Distance from the city centre",
        labels={
            "DIS": "Distance from the city centre to the house",
            "RM": "Number of rooms in the house",
        },
    )
    fig.update_layout(
        font_family="Rockwell",
        legend=dict(
            title=None, orientation="v", y=0.81, yanchor="bottom", x=0.855, xanchor="center"
        ),
    )
    return fig


parse_data()
clean_data(["CRIM", "B", "ZN", "CHAS", "TAX", "B", "INDUS", "LSTAT"])
meanRM = 7
for index, row in data.iterrows():
    data.at[index, "DIS"] = round1(row["DIS"])
    # data.at[index, "B"] = 100 * round(row["B"] / 100)
    data.at[index, "RM"] = round(row["RM"]) if not isnan(row["RM"]) else meanRM
    data.at[index, "NOX"] = round2(10 * row["NOX"]) / 10
    data.at[index, "PTRATIO"] = round1(row["PTRATIO"])
    data.at[index, "AGE"] = 10 * round1(row["AGE"] / 10)
    data.at[index, "RAD"] = round1(row["RAD"])
for index, row in data1.iterrows():
    data1.at[index, "RM"] = row["RM"] if not isnan(row["RM"]) else meanRM
nm = "md"
data = data.rename(
    {
        "DIS": f"{nm}DIS",
        "RM": f"{nm}RM",
        "NOX": f"{nm}NOX",
        "PTRATIO": f"{nm}PTRATIO",
        "AGE": f"{nm}AGE",
        "RAD": f"{nm}RAD",
    },
    axis="columns",
)
data = pd.concat([data1.drop(columns=["MEDV"]), data], axis=1)
mean_price = mean(data["MEDV"].tolist())
std_price = std(data["MEDV"].tolist())
median_price = median(data["MEDV"].tolist())
max_price = max(data["MEDV"].tolist())
min_price = min(data["MEDV"].tolist())
upper_limit = 30
t1 = f"Price < ${int(1000 * upper_limit)}"
t2 = f"${int(1000 * upper_limit)} < Price"
data_segmented = pd.DataFrame(
    columns=[
        "NOX",
        "RM",
        "AGE",
        "DIS",
        f"{nm}NOX",
        f"{nm}RM",
        f"{nm}AGE",
        f"{nm}DIS",
        "MEDV",
        "TYPE",
    ]
)
for index, row in data.iterrows():
    to_append = [
        row["NOX"],
        row["RM"],
        row["AGE"],
        row["DIS"],
        row[f"{nm}NOX"],
        row[f"{nm}RM"],
        row[f"{nm}AGE"],
        row[f"{nm}DIS"],
        row["MEDV"],
    ]
    if row["MEDV"] >= upper_limit:
        to_append.append(t2)
        data_segmented.loc[-1] = to_append
        data_segmented.index = data_segmented.index + 1
    else:
        to_append.append(t1)
        data_segmented.loc[-1] = to_append
        data_segmented.index = data_segmented.index + 1
# data_segmented.sort_values(by=["MEDV"])
data_segmented = data_segmented.sort_index()


def main():
    page = st.sidebar.selectbox('Choose the page',
                                ['About', 'Hypothesis', 'Key Metrics', 'Price Segments Comparison', 'Graphical Analysis of Data'])
    if page == 'About':
        st.title("Real estate analysis")
        st.subheader('by Aleksei Pankin')
        """[Dataset](https://www.kaggle.com/datasets/arslanali4343/real-estate-dataset "kaggle.com/datasets/arslanali4343/real-estate-dataset") concerns housing values in suburbs of Boston.  

Number of Instances: 506  
Attribute Information:
  
    CRIM -per capita crime rate by town

    NOX -nitric oxides concentration (parts per 10 million)

    RM -average number of rooms per dwelling

    AGE -proportion of owner-occupied units built prior to 1940

    DIS -weighted distances to five Boston employment centres
    
    RAD -index of accessibility to radial highways
    
    PTRATIO - pupil-teacher ratio by town 
    
    MEDV -Median value of owner-occupied homes in 1000's dollars"""

    elif page == 'Hypothesis':
        st.header('Hypothesis')
        """ Let's consider development of world and constantly increasing average quality of life as axiom. Then let's state that the better surroundings of the house the more valuable it is. For some people it's also important to be close to the centre of the city, so we will consider it as an argument for price increase too. What is important too is age of the house, as communications tear over time, its logical for new houses to be more developed overall rather than olds ones, so, consequently, more valuable too.  

So, putting it all in one let's state the hypothesis: 
> _House is more valuable if it has good accessibility to the city centre, a lot of rooms, is of moderate age, good air quality around._

Further let's observe the dataset and *prove* or *refute* that hypothesis"""

    elif page == 'Key Metrics':
        page_KM = st.sidebar.selectbox('Choose the contents', ['Data', 'Chart'])
        if page_KM == 'Data':
            st.header('Key Metrics')
            col1, col2, col3 = st.columns(3)
            col1.metric("mean house price", value='$' + str(numerize(1000 * mean_price)))
            col2.metric('max house price', value='$' + str(numerize(1000 * max_price)))
            col3.metric('min house price', value='$' + str(numerize(1000 * min_price)))
            col4, col5 = st.columns(2)
            col4.metric('median house price', value='$' + str(numerize(1000 * median_price)))
            col5.metric('standard house price deviation', value='$' + str(numerize(1000 * std_price)))
            st.write(data[['NOX', 'RM', 'DIS', 'AGE', 'RAD', 'PTRATIO', 'MEDV']].describe())
        elif page_KM == 'Chart':
            st.header('Key Metrics')
            st.write(plot_key_metrics())

    elif page == 'Price Segments Comparison':
        page_PSC = st.sidebar.selectbox('Choose the metrics',
                                        ['Nitric Oxides over Distance', 'Number of Rooms over Age',
                                         'Age over Distance', 'Nitric Oxides over Age',
                                         'Number of Rooms over Distance'])
        st.header('Price Segments Comparison')
        if page_PSC == 'Nitric Oxides over Distance':
            st.subheader(
                'Dependence of the concentration of nitric oxide around the house on the distance from the city center to the house')
            """It is expected that number of rooms in house grows as it gets farther from the city centre.  
The graph below shows dependency of number of rooms in the house on distance from the city centre."""
            st.write(plot_psc_nox_dis())
            """ From the graph we can conclude that average number of rooms in expensive houses remain constant and equal to roughly 7.2 rooms in house, while average number of rooms in average/low priced houses increases as distance to the city centre from the house grows."""

        elif page_PSC == 'Number of Rooms over Age':
            st.subheader('Dependence of number of rooms in the house on its Age')
            """It is expected that in houses of higher price number of rooms is greater rather than in houses of average/low price.  
The graph below shows dependency of number of rooms in the house on its age. """
            st.write(plot_psc_rm_age())
        elif page_PSC == 'Age over Distance':
            st.subheader('Dependence of age of the house on its distance from the center of the city')
            """It is expected that number of rooms in house grows as it gets farther from the city centre.  
The graph below shows dependency of number of rooms in the house on distance from the city centre."""
            st.write(plot_psc_age_dis())
            """From the graph we can conclude that average number of rooms in expensive houses remains constant and equal to roughly 7.2 rooms in house, while average number of rooms in average/low priced houses increases as distance to the city centre from the house grows."""
        elif page_PSC == 'Nitric Oxides over Age':
            st.subheader('Dependence of nitric oxides concentration around the house on its age')
            """It is expected that nitric oxides concentration is lower around expensive houses rather than around average/low priced houses.  
The graph below shows dependency of nitric oxides concentration around house on its age."""
            st.write(plot_psc_nox_age())
            """From the graph we can conclude that on average expensive houses of the same age as average/low priced houses have better air environment for all ages varying."""
        elif page_PSC == 'Number of Rooms over Distance':
            st.subheader('Dependence of number of rooms in the house on its distance from the city centre')
            """It is expected that number of rooms in house grows as it gets farther from the city centre.  
The graph below shows dependency of number of rooms in the house on distance from the city centre."""
            st.write(plot_psc_rm_dis())
            """From the graph we can conclude that average number of rooms in expensive houses remain constant and equal to roughly 7.2 rooms in house, while average number of rooms in average/low priced houses increases as distance to the city centre from the house grows."""

    elif page == 'Graphical Analysis of Data':
        st.header('Graphical Analysis of Data')


if __name__ == "__main__":
    main()
