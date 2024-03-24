from data.transforms import clean_text
import pytest


@pytest.mark.parametrize(
    "text, clean",
    [
        (
            "Eli Roth is making a movie based on the game,  . . bit.ly/2wv4KHM",
            "Eli Roth is making a movie based on the game,  . .",
        ),
        (
            "@GearboxOfficial sooooo....when we getting that borderlands 3 xbox one x patch that fixes console shutting down???",
            "sooooo....when we getting that borderlands 3 xbox one x patch that fixes console shutting down???",
        ),
        (
            "@ GearboxOfficial sooooo.... if we get that Borderlands 3 xbox an x patch that fixes console shutdown???",
            "sooooo.... if we get that Borderlands 3 xbox an x patch that fixes console shutdown???",
        ),
        (
            "Sunday Funday! Enjoying Borderlands 3 twitch.tv/mad_man31",
            "Sunday Funday! Enjoying Borderlands 3",
        ),
        (
            "Sunday Funday! Enjoying Borderlands 3 twitch.tv / mad _ man31",
            "Sunday Funday! Enjoying Borderlands 3 mad man31",
        ),
        (
            "<unk> again",
            "again"
        ),
        (
            "<unk>",
            ""
        )
    ],
)
def test_clean_text(text, clean):
    assert clean_text(text) == clean
