"""telegram_bot.py.

User can send youtube url, bot replies with a summary of top negative comments.

run bot locally:
    ipy telegram_bot.py -i -- --dryrun
    ipy telegram_bot.py -i --

"""
import argparse
import logging
import os
import re
import traceback

from dotenv import load_dotenv
from rarc_utils.log import loggingLevelNames, setup_logger
from rarc_utils.telegram_bot import (delete_conv_msgs, get_handler_docstrings,
                                     toEscapeMsg)
from telegram import Update
from telegram.ext import (CallbackContext, CommandHandler, Filters,
                          MessageHandler, PicklePersistence, Updater)
from view_negatives import extract_negative_comments
from view_negatives import main as negatives_main
from yt_comments_analyzer import main as yt_main

log_fmt = "%(asctime)s - %(module)-16s - %(lineno)-4s - %(funcName)-16s - %(levelname)-7s - %(message)s"  # name
logger = setup_logger(
    cmdLevel=logging.INFO, saveFile=0, savePandas=0, jsonLogger=0, color=1, fmt=log_fmt
)  # URGENT WARNING DEBUG

load_dotenv()
TOKEN = os.environ.get("TELEGRAM_TOKEN")

# setup updating together with our telegram api token
pickler = PicklePersistence("persistance.pickle", store_callback_data=True)
updater = Updater(
    TOKEN,
    workers=4,
    persistence=pickler,
    use_context=True,
)

# dispatcher to register handlers
dp = updater.dispatcher

# recognizing any url format sent to the bot
url_regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|"  # ...or ipv4
    r"\[?[A-F0-9]*:[A-F0-9:]+\]?)"  # ...or ipv6
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


#####################
####    handlers
#####################


def start(update: Update, _: CallbackContext):
    """Send welcome message."""
    update.message.reply_text(
        "Welcome YouTube comments analyzer bot! \n\
        Send me any YouTube url, and I will analyze the comments for you. \n\
        For instructions on how to run this bot type /help"
    )


def help_(update: Update, _: CallbackContext):
    """Show a list of available commands."""
    ALL_HANDLERS = list(get_handler_docstrings(dp, sortAlpha=True).keys())
    # send a message once the command /hello is keyed
    # update.message.reply_text(f"{dir(context)=}")
    commands = "/" + "\n/".join(ALL_HANDLERS)
    update.message.reply_text(f"available commands: \n\n{commands}")


def help_more(update: Update, _: CallbackContext):
    """Show larger help message explaining the functionality of every command."""
    msg = "this bot supports following functionality: "

    return update.message.reply_html(toEscapeMsg(msg))


def error(update: Update, context: CallbackContext):
    """Log traceback for any internal bot errors."""
    logger.warning(
        'Update "%s" \n\ncaused error "%s \n%s"',
        update,
        context.error,
        traceback.format_exc(),
    )


def make_bold(x):
    return f"<b>{x}</b>" if not isinstance(x, str) or not x.startswith("<b>") else x


##########################
####    general handlers
##########################

match_any_url1 = re.compile(r"https://(.+)\n")
match_any_url2 = re.compile(r"https://(.+)$")

SCORE_COL = "Sentiment Score"
TEXT_COL = "Original Comment Text trunc"
# TEXT_COL = "Cleaned Comment Text trunc"

MAX_COMMENT_LEN = 300
MAX_TG_MSG_LEN = 4096


def text(update: Update, _: CallbackContext):
    """General method that responds to user text input."""
    res = match_any_url1.search(update.message.text) or match_any_url2.search(
        update.message.text
    )

    # todo: use cache for frequently analyzed videos
    # or only cache last video for now
    if res is not None:
        url = res.group(1)

        # run yt_comments_analyser
        yt_main(["--url", url])
        df = negatives_main()
        df = extract_negative_comments(df, maxstrlen=250)

        sel_df = df[[SCORE_COL, TEXT_COL]].head(MAX_COMMENT_LEN)
        recs = sel_df.to_records(index=False)

        msg = "\n\n".join(map(lambda x: "score={:.2f} \n{}".format(*x), recs))

        logger.info(f"print df: \n{df.head(5)}")

        # truncate message if it still too long
        msglen = len(msg)
        logger.info(f"{msglen=:,}")
        if msglen > 4096:
            logger.warning(f"msglen too big, truncating it")
            msg = msg[:MAX_TG_MSG_LEN]

        update.message.reply_text(msg)

    else:
        update.message.reply_text("Send me a valid YouTube url for comments analysis")


def main():
    """Describe main functionality."""
    # add command handlers for different command
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help_more", help_more))
    dp.add_handler(CommandHandler("help", help_))

    ##############################
    # other handlers below
    ##############################

    # add an handler for normal text (not commands)
    dp.add_handler(MessageHandler(Filters.text, text))

    # error logging, disable it if you want to debug interactively
    dp.add_error_handler(error)

    # delete all temporary message from previous chats that should be deleted
    delete_conv_msgs(dp)

    # start the bot
    updater.start_polling()

    # set the bot to run until you force it to stop
    updater.idle()


class ArgParser:
    """Create CLI parser."""

    @staticmethod
    def _create_parser():
        return argparse.ArgumentParser()

    @classmethod
    def get_parser(cls):
        """Add args to CLI parser."""
        CLI = cls._create_parser()

        CLI.add_argument(
            "-v",
            "--verbosity",
            type=str,
            default="info",
            help=f"choose debug log level: {', '.join(loggingLevelNames())}",
        )
        CLI.add_argument(
            "-l",
            "--local",
            action="store_true",
            help="connecting to local or global mysql server. make sure --network=host is enabled for the container",
        )
        CLI.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Only load browser and login, do nothing else",
        )

        return CLI


if __name__ == "__main__":

    parser = ArgParser.get_parser()
    args = parser.parse_args()

    if not args.dryrun:
        main()
