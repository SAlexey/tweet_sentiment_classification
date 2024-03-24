from data.utils import dictwrap, batched


def test_dictwrap():

    @dictwrap
    def f(x):
        return x

    assert f(1) == 1
    assert f(1, key="x") == {"x": 1}


def test_batched():

    @batched
    def f(x):
        return x

    assert f(1) == 1
    assert f([1, 2]) == [1, 2]


def test_dictwrap_batched():

    @dictwrap
    @batched
    def f(x):
        return x

    assert f(1) == 1
    assert f(1, key="x") == {"x": 1}
    assert f([1, 2]) == [1, 2]
    assert f([1, 2], key="x") == {"x": [1, 2]}
