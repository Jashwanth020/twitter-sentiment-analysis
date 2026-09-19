import pytest
from app import clean_text, preprocess_text

def test_clean_text_urls():
    text = "Check out this link https://example.com"
    result = clean_text(text)
    assert "http" not in result
    assert "link" in result

def test_clean_text_mentions():
    text = "Hello @username how are you?"
    result = clean_text(text)
    assert "@username" not in result
    assert "hello" in result

def test_clean_text_rt():
    text = "RT @username: This is a retweet."
    result = clean_text(text)
    assert "rt" not in result
    assert "retweet" in result

def test_preprocess_text():
    text = "The dogs are running fast!"
    result = preprocess_text(text)
    # running might stay running, but dogs should become dog
    assert "dog" in result
    assert "dogs" not in result
