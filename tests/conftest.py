import pytest

from rag_agent.pipeline import RAGPipeline
from rag_agent.types import Document


@pytest.fixture(scope="session")
def sample_pipeline() -> RAGPipeline:
    pipe = RAGPipeline()
    pipe.ingest_directory()
    return pipe


@pytest.fixture
def tiny_docs() -> list[Document]:
    return [
        Document("cats", "Cats", "# Cats\n\nCats purr when content. Cats hunt mice at night."),
        Document("dogs", "Dogs", "# Dogs\n\nDogs bark at strangers. Dogs fetch sticks happily."),
        Document("birds", "Birds", "# Birds\n\nBirds sing at dawn. Many birds migrate south in winter."),
    ]
