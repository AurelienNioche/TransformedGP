import pandas as pd
import numpy as np


def get_data(data_file="data_character_meaning.csv"):

    df = pd.read_csv(data_file, index_col=0,
                     dtype={"success": "boolean"},
                     parse_dates=['ts_display', 'ts_reply'])

    df.sort_values(["user", "ts_display"], inplace=True)

    # Keep only users from the last experiment (domain => 'active.fi')
    # and that did it until the end
    # (6 training session + 1 evaluation  session for each teacher
    #  = 14 sessions)
    df.drop(df[(df.domain != "active.fi") | (df.n_session_done != 14)].index,
            inplace=True)

    # Convert timestamps into seconds
    beginning_history = pd.Timestamp("1970-01-01", tz="UTC")
    df["timestamp"] = \
        (df["ts_reply"] - beginning_history).dt.total_seconds().values

    # Copy actual item ID in a new column
    df["item_id"] = df.item
    # Create new ids starting from zero
    for i, i_id in enumerate(df.item_id.unique()):
        df.loc[df.item_id == i_id, 'item'] = i

    # Total number of user
    n_u = len(df.user.unique())

    # Number of observations per user
    n_o_by_u = np.zeros(shape=n_u, dtype=int)
    for u, (user, user_df) in enumerate(df.groupby("user")):
        # Do not count first presentation
        n_o_by_u[u] = len(user_df) - len(user_df.item.unique())

        # Total number of observation
    n_obs = n_o_by_u.sum()

    # Replies (0: error, 1: success)
    y = np.zeros(shape=n_obs, dtype=int)
    # Time elapsed since the last presentation of the same item (in seconds)
    d = np.zeros(shape=n_obs, dtype=float)
    # Number of repetition (number of presentation - 1)
    r = np.zeros(shape=n_obs, dtype=int)
    # Item ID
    w = np.zeros(shape=n_obs, dtype=int)
    # User ID
    u = np.zeros(shape=n_obs, dtype=int)

    # Fill the containers `y`, `x`, `r`, `w`, `u`
    idx = 0
    for i_u, (user, user_df) in enumerate(df.groupby("user")):

        # Extract data from user `u`
        user_df = user_df.sort_values(by="timestamp")
        seen = user_df.item.unique()
        w_u = user_df.item.values
        ts_u = user_df.timestamp.values
        y_u = user_df.success.values

        # Initialize counts of repetition for each words at -1
        counts = {word: -1 for word in seen}
        # Initialize time of last presentation at None
        last_pres = {word: None for word in seen}

        # Number of observations for user `u` including first presentations
        n_obs_u_incl_first = len(user_df)

        # Number of repetitions for user `u`
        r_u = np.zeros(n_obs_u_incl_first)
        # Time elapsed since last repetition for user `u`
        d_u = np.zeros(n_obs_u_incl_first)

        # Loop over each entry for user `u`:
        for i in range(n_obs_u_incl_first):

            # Get info for iteration `i`
            word = w_u[i]
            ts = ts_u[i]
            r_u[i] = counts[word]

            # Compute time elasped since last presentation
            if last_pres[word] is not None:
                d_u[i] = ts - last_pres[word]

            # Update count of repetition
            counts[word] += 1
            # Update last presentation
            last_pres[word] = ts

        # Keep only observations that are not the first presentation of an item
        to_keep = r_u >= 0
        y_u = y_u[to_keep]
        r_u = r_u[to_keep]
        w_u = w_u[to_keep]
        d_u = d_u[to_keep]

        # Number of observations for user `u` excluding first presentations
        n_obs_u = len(y_u)

        # Fill containers
        y[idx:idx + n_obs_u] = y_u
        d[idx:idx + n_obs_u] = d_u
        w[idx:idx + n_obs_u] = w_u
        r[idx:idx + n_obs_u] = r_u
        u[idx:idx + n_obs_u] = i_u

        # Update index
        idx += n_obs_u

    n_w = len(np.unique(w))
    n_o_max = n_o_by_u.max()
    n_o_min = n_o_by_u.min()
    print("number of user", n_u)
    print("number of items", n_w)
    print("total number of observations (excluding first presentation)", n_obs)
    print("minimum number of observation for a single user", n_o_min)
    print("maximum number of observation for a single user", n_o_max)

    df = pd.DataFrame({
        'u': u,  # User ID
        'w': w,  # Item ID
        'd': d,
        # Time elapsed since the last presentation of the same item
        # (in seconds)
        'r': r,  # Number of repetition (number of presentation - 1)
        'y': y  # Replies (0: error, 1: success)
    })

    return df
