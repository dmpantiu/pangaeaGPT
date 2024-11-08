# Custom UI Styles
CUSTOM_UI = """
<style>
    :root {
        /* Main colors (lighter versions) */
        --primary-teal: rgb(67, 163, 151);          /* Lighter version of #129187 */
        --primary-teal-light: rgba(67, 163, 151, 0.1);
        --neutral-gray: rgb(220, 220, 220);         /* Lighter version of #c0c0c0 */
        --primary-blue: rgb(82, 142, 198);          /* Lighter version of #337ab7 */
        --primary-blue-light: rgba(82, 142, 198, 0.1);
        --dark-blue: rgb(65, 105, 145);             /* Lighter version of #23527c */
        --cream-bg: rgba(218,6,18, 0.05);             /* #fcf8e3 */
        
        /* Added light red accents */
        --accent-red: rgb(235, 108, 108);           /* Light red */
        --accent-red-light: rgba(235, 108, 108, 0.1);
        
        /* Base colors */
        --white: #ffffff;
        --text-dark: rgb(60, 60, 60);
    }

    /* Global Styles */
    .stApp {
        background-color: var(--white);
        color: var(--text-dark);
    }

    /* Headers */
    h1, h2, h3 {
        color: var(--dark-blue);
        border-bottom: 2px solid var(--primary-teal-light);
    }

    /* Chat Messages */
    .stChatMessage {
        background: var(--primary-blue-light);
        border-left: 3px solid var(--primary-blue);
    }

    /* Buttons */
    .stButton > button {
        background: var(--white);
        color: var(--primary-teal);
        border: 1px solid var(--primary-teal);
    }
    
    .stButton > button:hover {
        background: var(--primary-teal);
        color: var(--white);
    }

    /* Fixed Button */
    .fixed-button button {
        background: var(--primary-blue);
        color: var(--white);
    }

    /* Sidebar */
    [data-testid=stSidebar] {
        background-color: var(--cream-bg);
        color: var(--text-dark);
        border-right: 1px solid var(--neutral-gray);
    }

    [data-testid=stSidebar] .stSelectbox label {
        color: var(--text-dark);
    }

    /* Input Fields */
    .stTextInput > div > div > input {
        background: var(--white);
        border: 1px solid var(--neutral-gray);
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 1px var(--primary-blue-light);
    }

    /* Alerts and Messages */
    .stAlert {
        background: var(--primary-teal-light);
        border-left: 3px solid var(--primary-teal);
    }

    /* Error messages or warnings */
    .stAlert.error {
        background: var(--accent-red-light);
        border-left: 3px solid var(--accent-red);
    }

    /* DataFrame */
    .stDataFrame {
        border: 1px solid var(--neutral-gray);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--white);
        border: 1px solid var(--neutral-gray);
    }

    /* Checkbox */
    .stCheckbox > label > div[role="checkbox"] {
        border-color: var(--primary-teal);
    }

    .stCheckbox > label > div[role="checkbox"][aria-checked="true"] {
        background-color: var(--primary-teal);
    }

    /* Search bar */
    .stSearchInput > div > div > input {
        border: 1px solid var(--primary-blue);
    }

    .stSearchInput > div > div > input:focus {
        border-color: var(--primary-teal);
        box-shadow: 0 0 0 1px var(--primary-teal-light);
    }

    /* Secondary buttons */
    [data-testid="stButton"] > button[kind="secondary"] {
        background: var(--white);
        color: var(--primary-blue);
        border: 1px solid var(--primary-blue);
    }
    
    [data-testid="stButton"] > button[kind="secondary"]:hover {
        background: var(--primary-blue);
        color: var(--white);
    }

    /* Delete or cancel actions */
    .delete-button > button {
        color: var(--accent-red);
        border-color: var(--accent-red);
    }

    .delete-button > button:hover {
        background: var(--accent-red);
        color: var(--white);
    }
</style>
"""


# Constants
SYSTEM_ICON = "img/11111111.png"
USER_ICON = "img/2222222.png"