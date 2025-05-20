import datetime
from app import *
import altair as alt
import numpy as np
import pandas as pd

# Show app title and description.
st.set_page_config(page_title="Support tickets", page_icon="🎫")
st.title("🎫 Support tickets Admin Section")
# st.write(
#     """
#     This app shows how you can build an internal tool in Streamlit. Here, we are
#     implementing a support ticket workflow. The user can create a ticket, edit
#     existing tickets, and view some statistics.
#     """
# )

# Create a random Pandas dataframe with existing tickets.
if "df" not in st.session_state:

    # Set seed for reproducibility.
    np.random.seed(42)

    # Make up some fake issue descriptions.
    issue_descriptions = ["hii"]

    # Generate the dataframe with 100 rows/tickets.
    data = {"ID": [f"TICKET-{1}"],
        "Issue": issue_descriptions,
        "Status": ["Open"],
        "Department": ["IT"],
        "Date Submitted": datetime.datetime.now().strftime("%m-%d-%Y"),
            }
    df = pd.DataFrame(data)

    # Save the dataframe in session state (a dictionary-like object that persists across
    # page runs). This ensures our data is persisted when the app updates.
    st.session_state.df = df


# Show a section to add a new ticket.


# We're adding tickets via an `st.form` and some input widgets. If widgets are used
# in a form, the app will only rerun once the submit button is pressed.
# Make a dataframe for the new ticket and append it to the dataframe in session
# state.
recent_ticket_number = int(max(st.session_state.df.ID).split("-")[1])
today = datetime.datetime.now().strftime("%m-%d-%Y")
df_new = pd.DataFrame(
    [
        {
            "ID": f"TICKET-{recent_ticket_number+1}",
            "Issue": st.session_state['tickets'],
            "Status": "Open",
            "Department": st.session_state['department_value'],
            "Date Submitted": today,
        }
    ]
)

# Show a little success message.
st.write("Ticket submitted! Here are the ticket details:")
st.dataframe(df_new, use_container_width=True, hide_index=True)
st.session_state.df = pd.concat([df_new, st.session_state.df], axis=0)

# Show section to view and edit existing tickets in a table.
st.header("Existing tickets")
st.write(f"Number of tickets: `{len(st.session_state.df)}`")

st.info(
    "You can edit the tickets by double clicking on a cell. Note how the plots below "
    "update automatically! You can also sort the table by clicking on the column headers.",
    icon="✍️",
)

# Show the tickets dataframe with `st.data_editor`. This lets the user edit the table
# cells. The edited data is returned as a new dataframe.
edited_df = st.data_editor(
    st.session_state.df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.SelectboxColumn(
            "Status",
            help="Ticket status",
            options=["Open", "In Progress", "Closed"],
            required=True,
        ),
        "Department": st.column_config.SelectboxColumn(
            "Department",
            help="Department",
            options=["HR", "IT", "Transport"],
            required=True,
        ),
    },
    # Disable editing the ID and Date Submitted columns.
    disabled=["ID", "Date Submitted"],
)

# Show some metrics and charts about the ticket.
st.header("Statistics")

# Show metrics side by side using `st.columns` and `st.metric`.
col1, col2, col3 = st.columns(3)
num_open_tickets = len(st.session_state.df[st.session_state.df.Status == "Open"])
col1.metric(label="Number of open tickets", value=num_open_tickets, delta=10)
col2.metric(label="First response time (hours)", value=5.2, delta=-1.5)
col3.metric(label="Average resolution time (hours)", value=16, delta=2)

# Show two Altair charts using `st.altair_chart`.
st.write("")
st.write("##### Ticket status per month")
status_plot = (
    alt.Chart(edited_df)
    .mark_bar()
    .encode(
        x="month(Date Submitted):O",
        y="count():Q",
        xOffset="Status:N",
        color="Status:N",
    )
    .configure_legend(
        orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
    )
)
st.altair_chart(status_plot, use_container_width=True, theme="streamlit")

st.write("##### Current ticket Departments")
Departmental_plot = (
    alt.Chart(edited_df)
    .mark_arc()
    .encode(theta="count():Q", color="Department:N")
    .properties(height=300)
    .configure_legend(
        orient="bottom", titleFontSize=14, labelFontSize=14, titlePadding=5
    )
)
st.altair_chart(Departmental_plot, use_container_width=True, theme="streamlit")
