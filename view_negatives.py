"""view_negatives.py.

Open dataset created by yt_comments_analyzer.py and print top negative comments of a YouTube video.
"""

import pandas as pd
from colorama import Fore, Style

FILE = "output/res.csv"

TEXT_COL = "Original Comment Text"
# TEXT_COL = "Cleaned Comment Text"
SCORE_COL = "Sentiment Score"
SENTIMENT = "Sentiment"
NEGATIVE = "Negative"


def print_top_negative_comments(
    df: pd.DataFrame,
    textCol=TEXT_COL,
    sentimentCol=SCORE_COL,
    onlyNegative=True,
    top=20,
) -> None:
    """Print top negative comments from dataframe.

    Dataframe was created b y running yt_comments_analyzer.py, example:
        python yt_comments_analyzer.py -u \
            https://www.youtube.com/watch?v=XA2WjJbmmoM
        
        print_top_negative_comments(df)
    """
    # use only negative comments
    if onlyNegative:
        df = df[df[SENTIMENT] == NEGATIVE].copy()

    assert (
        not df.empty
    ), "please pass non-empty dataframe, or toggle `onlyNegative=False`"

    for ix, (_, row) in enumerate(df.head(top).iterrows()):
        sentiment = row.loc[sentimentCol]
        print(f"{Fore.RED}negative comment #{ix} {Style.RESET_ALL} {sentiment=:.3f}")
        print(" ", row.loc[textCol], "\n")


def main() -> None:
    """Contains main functionality."""
    df = pd.read_csv(FILE)

    df.sort_values(SCORE_COL, ascending=True, inplace=True)

    print_top_negative_comments(df, onlyNegative=1, top=40)


if __name__ == "__main__":
    main()
