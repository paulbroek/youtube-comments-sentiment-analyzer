"""transformers.py.

Use HuggingFace transformers for sentiment analysis
"""
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
data = [
    "I love you",
    "I hate you",
    "I dont like that you are so amazing",
    "You are not a bad person, not at all.",
    "You are an average person, right."
]
sentiment_pipeline(data)
