import pytest

from cybersec_priority_bot.criticality_classifier import CriticalityClassifier

@pytest.fixture(scope="session")
def criticality_classifier():
    yield CriticalityClassifier()

