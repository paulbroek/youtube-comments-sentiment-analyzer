r"""view_negatives.py.

Open dataset created by yt_comments_analyzer.py and print top negative comments of a YouTube video.

Usage:
    export GCLOUD_CONFIG_FILE=/home/paul/repos/gcloud-utils/gcloud_utils/config/config.yaml && \                                                                                                                                          1 ⨯
    export GOOGLE_APPLICATION_CREDENTIALS="/home/paul/Downloads/service-account-file.json"

    python yt_comments_analyzer.py -u \
        https://www.youtube.com/watch?v=XA2WjJbmmoM
    
    ipy view_negatives.py
"""

import numpy as np
import pandas as pd
from colorama import Fore, Style  # type: ignore[import]

FILE = "output/res.csv"

TEXT_COL = "Original Comment Text"
# TEXT_COL = "Cleaned Comment Text"
SCORE_COL = "Sentiment Score"
SENTIMENT = "Sentiment"
NEGATIVE = "Negative"


def extract_negative_comments(df: pd.DataFrame, maxstrlen=None) -> pd.DataFrame:
    df = df[df[SENTIMENT] == NEGATIVE].copy()

    # optionally truncate str columns
    if maxstrlen is not None:
        text_cols = df.filter(regex=".+Text$").columns
        for col in text_cols:
            df[f"{col} len"] = df[col].map(len)
            trunccol = f"{col} trunc"
            df[trunccol] = df[col].str.slice(0, maxstrlen)
            df[trunccol] = np.where(
                df[f"{col} len"] > maxstrlen, df[trunccol], df[trunccol] + "..."
            )

    return df


def format_negative_comments(
    df: pd.DataFrame,
    textCol=TEXT_COL,
    sentimentCol=SCORE_COL,
    onlyNegative=True,
    maxstrlen=None,
    top=20,
) -> str:
    """Print top negative comments from dataframe.

    Usage:
        format_negative_comments(df, onlyNegative=1, top=40)

    """
    # use only negative comments
    if onlyNegative:
        df = extract_negative_comments(df, maxstrlen=maxstrlen)

    assert (
        not df.empty
    ), "please pass non-empty dataframe, or toggle `onlyNegative` to False"

    output = ""

    for ix, (_, row) in enumerate(df.head(top).iterrows()):
        sentiment = row.loc[sentimentCol]
        output += (
            f"{Fore.RED}negative comment #{ix} {Style.RESET_ALL} {sentiment=:.3f}\n"
        )
        output += f" {row.loc[textCol]}\n"

    return output


def main() -> pd.DataFrame:
    """Load comments and print negative comments."""
    df = pd.read_csv(FILE, lineterminator="\n")

    df.sort_values(SCORE_COL, ascending=True, inplace=True)

    return df


if __name__ == "__main__":
    df_ = main()
    print(format_negative_comments(df_, onlyNegative=1, top=40, maxstrlen=50))
