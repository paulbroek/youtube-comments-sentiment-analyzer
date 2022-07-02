r"""view_negatives.py.

Open dataset created by yt_comments_analyzer.py and print top negative comments of a YouTube video.

Usage:
    export GCLOUD_CONFIG_FILE=/home/paul/repos/gcloud-utils/gcloud_utils/config/config.yaml && \                                                                                                                                          1 ⨯
    export GOOGLE_APPLICATION_CREDENTIALS="/home/paul/Downloads/service-account-file.json"

    python yt_comments_analyzer.py -u \
        https://www.youtube.com/watch?v=XA2WjJbmmoM
    
    ipy view_negatives.py
"""

import pandas as pd
from colorama import Fore, Style  # type: ignore[import]

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

    Usage:
        print_top_negative_comments(df, onlyNegative=1, top=40)

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
    """Load comments and print negative comments."""
    df = pd.read_csv(FILE, lineterminator='\n')

    df.sort_values(SCORE_COL, ascending=True, inplace=True)

    print_top_negative_comments(df, onlyNegative=1, top=40)


if __name__ == "__main__":
    main()
