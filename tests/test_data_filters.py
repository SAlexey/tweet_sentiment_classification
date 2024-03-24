from data.filters import keep_text, keep_label
import pytest


def test_keep_text():

    texts = ["foobar", "foobar" * 2, "foobar" * 3]

    # keep all texts without any constraints
    assert keep_text(texts) == [True, True, True]

    # keep foobar with min_length == len(foobar)
    min_length = len("foobar")
    assert keep_text(texts, min_length=min_length) == [True, True, True]

    # drop foobar with min_length > len(foobar)
    assert keep_text(texts, min_length=min_length + 1) == [False, True, True]


def test_keep_label():

    labels = ["foo", "bar", "bar", "baz", "baz"]
    exclude = ["foo"]
    include = ["bar", "baz"]

    # keep all labels without constraints
    assert keep_label(labels) == [True, True, True, True, True]

    # drop foo with exclude constraint
    assert keep_label(labels, exclude=exclude) == [False, True, True, True, True]

    # drop foo with include constraint
    assert keep_label(labels, include=include) == [False, True, True, True, True]
