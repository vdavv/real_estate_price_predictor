import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from numerize.numerize import numerize
from numpy import *
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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


def plot_linear_regression(x, y, ttpe="lowess", ttle="", ttle_x="", ttle_y=""):
    x_train, x_test, y_train, y_test = train_test_split(
        data[[x]], data[y], test_size=0.3, random_state=100
    )
    # splitting the dataset for training and testing
    slr = LinearRegression()
    slr.fit(x_train, y_train)
    # fitting the model
    # print("Intercept: ", slr.intercept_)
    # print("Coefficient: ", slr.coef_)
    y_pred_slr = slr.predict(x_test)
    # print(y_pred_slr)
    # prediction
    df = pd.concat(
        [
            x_test,
            y_test,
            pd.DataFrame(data=map(lambda x: [x], y_pred_slr), columns=["y_pred"]),
        ],
        axis=1,
        join="inner",
    )
    return px.scatter(
        df,
        x=x,
        y=y,
        trendline=ttpe,
        trendline_color_override="Red",
        title=ttle,
        marginal_y="box",
        marginal_x="violin",
        height=800, width=800,
        labels={x: ttle_x, y: ttle_y},
    )


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
                                ['About', 'Key indicators', 'Price Segments Comparison',
                                 'General Data Analysis', 'Linear Regressions', 'Conclusion'])
    if page == 'About':
        """
        # Real estate analysis
        ##### `>>> 'Aleksei Pankin'.iscreator()`
        ##### `True`
        """
        """---"""
        st.header('Hypothesis')
        """[Dataset](https://www.kaggle.com/datasets/arslanali4343/real-estate-dataset "kaggle.com/datasets/arslanali4343/real-estate-dataset") concerns housing values in suburbs of Boston.  
                """
        """ Let's consider development of world and constantly increasing average quality of life as axiom. Then let's state that the better surroundings of the house the more valuable it is. For some people it's also important to be close to the centre of the city, so we will consider it as an argument for price increase too. What is important too is age of the house, as communications tear over time, its logical for new houses to be more developed overall rather than olds ones, so, consequently, more valuable too.  

So, putting it all in one let's state the hypothesis: 
> _House is more valuable if it has good accessibility to the city centre, a lot of rooms, is of moderate age, good air quality around._

Further let's observe the dataset and *prove* or *refute* that hypothesis"""

    elif page == 'Key indicators':
        st.header('Key indicators')
        """---"""
        page_KM = st.sidebar.selectbox('Choose the contents', ['Data', 'Chart'])
        if page_KM == 'Data':
            col1, col2, col3 = st.columns(3)
            col1.metric("mean house price", value='$' + str(numerize(1000 * mean_price)))
            col2.metric('max house price', value='$' + str(numerize(1000 * max_price)))
            col3.metric('min house price', value='$' + str(numerize(1000 * min_price)))
            col4, col5 = st.columns(2)
            col4.metric('median house price', value='$' + str(numerize(1000 * median_price)))
            col5.metric('standard house price deviation', value='$' + str(numerize(1000 * std_price)))
            st.write(data[['NOX', 'RM', 'DIS', 'AGE', 'RAD', 'PTRATIO', 'MEDV']].describe())
            """
                ```
                NOX -nitric oxides concentration (parts per 10 million)

                RM -average number of rooms per dwelling
                
                DIS -weighted distances to five Boston employment centres
                
                AGE -proportion of owner-occupied units built prior to 1940
                
                RAD -index of accessibility to radial highways

                PTRATIO - pupil-teacher ratio by town 

                MEDV -Median value of owner-occupied homes in 1000's dollars
                ```"""
        elif page_KM == 'Chart':
            st.write(plot_key_metrics())

    elif page == 'Price Segments Comparison':
        st.header('Price Segments Comparison')

        """---"""
        st.subheader(
            'Dependence of the concentration of nitric oxide around the house on the distance from the city center to the house')
        """It is expected that number of rooms in house grows as it gets farther from the city centre.  
The graph below shows dependency of number of rooms in the house on distance from the city centre."""
        st.write(plot_psc_nox_dis())
        """ From the graph we can conclude that average number of rooms in expensive houses remain constant and equal to roughly 7.2 rooms in house, while average number of rooms in average/low priced houses increases as distance to the city centre from the house grows."""

        """---"""
        st.subheader('Dependence of number of rooms in the house on its Age')
        """It is expected that in houses of higher price number of rooms is greater rather than in houses of average/low price.  
The graph below shows dependency of number of rooms in the house on its age. """
        st.write(plot_psc_rm_age())

        """---"""
        st.subheader('Dependence of age of the house on its distance from the center of the city')
        """It is expected that number of rooms in house grows as it gets farther from the city centre.  
The graph below shows dependency of number of rooms in the house on distance from the city centre."""
        st.write(plot_psc_age_dis())
        """From the graph we can conclude that average number of rooms in expensive houses remains constant and equal to roughly 7.2 rooms in house, while average number of rooms in average/low priced houses increases as distance to the city centre from the house grows."""

        """---"""
        st.subheader('Dependence of nitric oxides concentration around the house on its age')
        """It is expected that nitric oxides concentration is lower around expensive houses rather than around average/low priced houses.  
The graph below shows dependency of nitric oxides concentration around house on its age."""
        st.write(plot_psc_nox_age())
        """From the graph we can conclude that on average expensive houses of the same age as average/low priced houses have better air environment for all ages varying."""

        """---"""
        st.subheader('Dependence of number of rooms in the house on its distance from the city centre')
        """It is expected that number of rooms in house grows as it gets farther from the city centre.  
The graph below shows dependency of number of rooms in the house on distance from the city centre."""
        st.write(plot_psc_rm_dis())
        """From the graph we can conclude that average number of rooms in expensive houses remain constant and equal to roughly 7.2 rooms in house, while average number of rooms in average/low priced houses increases as distance to the city centre from the house grows."""

    elif page == 'General Data Analysis':
        st.header('General Data Analysis')
        """---"""
        page_GAD = st.sidebar.selectbox('Choose the plot',
                                        ['Distance', 'Nitric Oxides', 'Age of House', 'Number of Rooms'])
        if page_GAD == 'Pairwise':
            st.pyplot(sns.pairplot(data1[["MEDV", "RM", "DIS", "AGE", "NOX"]]))
        elif page_GAD == 'Distance':
            st.subheader('Plots involving Distance from the city centre to the house')
            """It is expected that most expensive houses are located near to the city centre.  
The graph below shows dependency of distance from city centre to house on price of house."""
            st.write(px.histogram(
                data[["MEDV", f"{nm}DIS"]],
                x="MEDV",
                color=f"{nm}DIS",
                marginal="box",
                title="Distance to the city centre over Price of house",
                height=800,
                width=800,
                histfunc="count",
                labels={
                    "MEDV": "Price of House in 1000 USD",
                    "y": "Number of houses",
                    f"{nm}DIS": "Distance",
                },
            ).update_layout(yaxis_title="Number of houses"))
            """From the graph we can conclude that on average closest to the centre houses are rather extremeley low priced or high priced.  """
            """---"""
            """The graph below shows dependency of nitric oxides concentration on distance of house from the city centre."""
            st.write(px.histogram(
                data[[f"{nm}NOX", f"{nm}DIS"]],
                x=f"{nm}DIS",
                color=f"{nm}NOX",
                marginal="box",
                title="Nitric oxides concentration over Distance",
                height=800,
                width=800,
                labels={
                    f"{nm}DIS": "Distance from the city centre to the house",
                    f"{nm}NOX": "NOX",
                },
            ).update_layout(yaxis_title="Number of houses"))

        elif page_GAD == 'Number of Rooms':
            st.subheader('Plots involving Number of Rooms in the house')
            """It is expected that the more rooms in a house, the higher its price.  
The graph below shows dependency of number of rooms on price of house."""
            st.write(
                px.histogram(
                    data[["MEDV", f"{nm}RM"]],
                    x="MEDV",
                    color=f"{nm}RM",
                    marginal="box",
                    title="Number of Rooms over Price of house",
                    height=800,
                    width=800,
                    labels={"MEDV": "Price of house in 1000 USD", f"{nm}RM": "№ Rooms"},
                ).update_layout(yaxis_title="Number of houses"))
            """From the graph we can conclude that on average the more house is expensive, the more rooms it has, as expected."""
            """---"""
            """The graph below shows dependency of age of house on number of rooms in it."""
            st.write(px.histogram(
                data[[f"{nm}AGE", f"{nm}RM"]],
                x=f"{nm}RM",
                color=f"{nm}AGE",
                title="Age of house over Number of rooms in it",
                height=800,
                width=800,
                labels={f"{nm}RM": "Number of rooms in the house", f"{nm}AGE": "Age"},
            ).update_layout(yaxis_title="Number of houses"))
            """From the graph we can conclude that 90% of houses have 6-7 rooms, that old houses on average tend to have less rooms than new and that houses with the smallest amount of room are almost all old, from all of the stated above we can state that average house has more rooms over time."""

        elif page_GAD == 'Nitric Oxides':
            st.subheader('Plots involving Nitric Oxides Concentration around the house')
            """It is expected that the lower the concentration of nitric oxides, the higher the price of the house.  
The graph below shows dependancy of nitric oxides concentration on price of house."""
            st.write(px.histogram(
                data[["MEDV", f"{nm}NOX"]],
                x="MEDV",
                color=f"{nm}NOX",
                marginal="box",
                title="Nitric oxides concentration over Price of house",
                height=800,
                width=800,
                labels={
                    "MEDV": "Price of house in 1000 USD",
                    f"{nm}NOX": "NOX",
                },
            ).update_layout(yaxis_title="Number of houses"))
            """From the graph we can conclude that on average the more house is expensive, the cleaner air around it."""
            """---"""
            """The graph below shows dependency of nitric oxides concentration on distance of house from the city centre. From the graph we can conclude that on average the more house is distanced from the city centre, the lower the concentration of nitric oxides in air around it."""
            st.write(px.histogram(
                data[[f"{nm}NOX", f"{nm}DIS"]],
                x=f"{nm}DIS",
                color=f"{nm}NOX",
                marginal="box",
                title="Nitric oxides concentration over Distance",
                height=800,
                width=800,
                labels={
                    f"{nm}DIS": "Distance from the city centre to the house",
                    f"{nm}NOX": "NOX",
                },
            ).update_layout(yaxis_title="Number of houses"))

        elif page_GAD == 'Age of House':
            st.subheader('Plots involving Age of house')
            """It is expected that the newer the house, the higher its price.  
The graph below shows dependency of age of house on its price."""
            st.write(px.histogram(
                data[["MEDV", f"{nm}AGE"]],
                x="MEDV",
                color=f"{nm}AGE",
                marginal="box",
                title="Age of house over its Price",
                height=800,
                width=800,
                labels={"MEDV": "Price of house in 1000 USD", f"{nm}AGE": "Age"},
            ).update_layout(yaxis_title="Number of houses"))
            """From the graph we can conclude that on average old houses are rather extremely cheap or expensive, while the age of average priced house varies from 10 to 90 years in almost equal proportions."""
            """---"""
            """The graph below shows dependency of age of house on number of rooms in it. From the graph we can conclude that 90% of houses have 6-7 rooms, that old houses on average tend to have less rooms than new and that houses with the smallest amount of room are almost all old, from all of the stated above we can state that average house has more rooms over time"""
            st.write(px.histogram(
                data[[f"{nm}AGE", f"{nm}RM"]],
                x=f"{nm}RM",
                color=f"{nm}AGE",
                title="Age of house over Number of rooms in it",
                height=800,
                width=800,
                labels={f"{nm}RM": "Number of rooms in the house", f"{nm}AGE": "Age"},
            ).update_layout(yaxis_title="Number of houses"))
    elif page == 'Linear Regressions':
        st.header('Linear Regressions')
        """---"""
        page_LR = st.sidebar.selectbox('Choose metrics',
                                       ['Distance', 'Nitric Oxides', 'Age of House', 'Number of Rooms'])

        if page_LR == 'Distance':
            st.subheader('Linear regressions involving Distance from the house to the city centre')
            """It is expected that the closer the house to the city centre, the higher its price  
The graph below shows dependency of distance from the house to the city centre on its price."""
            st.write(plot_linear_regression(
                "MEDV",
                "DIS",
                "ols",
                "Distance to the city centre over Price of house",
                "Price of house in 1000 USD",
                "Distance from the house to the city centre",
            ))
            """From the graph we can conclude that on average the more house is distanced from the city, the more valuable it is. Such a conclusion is actually quite contradictionary for me, but, well, statistics knows better. It may be so as people prefer to be further from the city to unite with nature and relax from urban hustle and bustle, as they anyway visit it almost every day for work."""
            """---"""
            """It is expected that concentration of nitric oxides drops as its gets farther from the city centre.
The graph below shows dependency of nitric oxides concentration in the air around the house on distance to the city centre from it."""
            st.write(plot_linear_regression(
                "DIS",
                "NOX",
                "ols",
                "Nitric oxides concentration over Distance",
                "Distance from the city centre to the house",
                "Nitric oxides concentration in the air around the house",
            ))
            """From the graph we can conclude that as expected the farther the house is from the city centre, the lower the concentration of oxides of air."""
        elif page_LR == 'Nitric Oxides':
            st.subheader('Linear regressions involving Nitric Oxides Concentration in the air around the house')
            """It is expected that the lower the concentration of nitric oxides, the higher the price of the house.  
The graph below shows dependency of nitric oxides concentration around the house on its price."""
            st.write(plot_linear_regression(
                "MEDV",
                "NOX",
                "ols",
                "Nitric oxides concentration over Price of house",
                "Price of house in 1000 USD",
                "Nitric oxides concentration in air around the house",
            ))
            """From the graph we can conclude that better the air surrounds the house the more the valuable the house. Well, that was actually obvious from the very beginning, but we proved it statistically, so that now we can be sure that it is so."""
            """---"""
            """It is expected that concentration of nitric oxides drops as its gets farther from the city centre.  
The graph below shows dependency of nitric oxides concentration in the air around the house on distance to the city centre from it."""
            st.write(plot_linear_regression(
                "DIS",
                "NOX",
                "ols",
                "Nitric oxides concentration over Distance",
                "Distance from the city centre to the house",
                "Nitric oxides concentration in the air around the house",
            ))
            """From the graph we can conclude that as expected the farther the house is from the city centre, the lower the concentration of oxides of air."""
        elif page_LR == 'Age of House':
            st.subheader('Linear regressions involving Age of house')
            """It is expected that the newer the house, the higher its price.  
The graph below shows dependency of age of house on its price."""
            st.write(plot_linear_regression(
                "MEDV",
                "AGE",
                "ols",
                "Age of house over its Price",
                "Price of house in 1000 USD",
                "Age of house",
            ))
            """From the graph we can conclude that that on average the newer the house, the more expensive it is. However, there are a lot of deviations from a trendline, so some exceptions should be considered. It may be caused by many effects, but on my sight the main are that old expensive houses can be a historical legacy and cheap houses may be a consequence of tradeoff of location and price, so that better location is preferred rather than quality of communications."""

        elif page_LR == 'Number of Rooms':
            st.subheader('Linear regressions involving Number of Rooms in the house')
            """It is expected that the more rooms in a house, the higher its price.  
The graph below shows dependency of room number on price."""
            st.write(plot_linear_regression(
                "MEDV",
                "RM",
                "ols",
                "Number of Rooms in house over its Price",
                "Price of house in 1000 USD",
                "Number of Rooms in house",
            ))
            """From the graph we can conclude that with good precision the more house is expensive, the more rooms it has. It may be so as wealthy people prefer bigger houses with more rooms as they can afford them, and consequently, expensive houses is built with more rooms than average houses."""

    elif page == 'Conclusion':
        st.header('Conclusion')
        """---"""
        """Well, as we can see above, most of my statements were confirmed, although, there are some that were refuted. Let's state them further:"""
        """> *House is more valuable if it is distanced from city centre, has clean air in its surroundings, a lot of rooms and in average of moderate age*"""
        """As we can see on the graphs, the air condisitons near house gets better as house is more distanced from the city, so that we can say that ecology is more important to people, rather than time to get to work. What comes to age, maybe, as I stated before, people tradeoff communications quality to better location, or for expensive houses live in historical legacy and doesn't want to move out for personal reasons."""
        """---"""
        """##### `>>> final_word()`"""
        """##### `Thanks for your attention!`"""


if __name__ == "__main__":
    main()
