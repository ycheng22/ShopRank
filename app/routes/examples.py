from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class DemoQuery(BaseModel):
    query: str
    locale: str
    description: str


DEMO_QUERIES = [
    DemoQuery(query="wireless earbuds", locale="en", description="Short, high-traffic"),
    DemoQuery(
        query="stainless steel insulated water bottle for hiking 32 oz",
        locale="en",
        description="Long, multi-attribute",
    ),
    DemoQuery(query="blouetooth speaker", locale="en", description="Misspelling"),
    DemoQuery(query="蓝牙耳机", locale="zh", description="Cross-lingual (Chinese)"),
    DemoQuery(
        query="casque sans fil", locale="fr", description="Cross-lingual (French)"
    ),
    DemoQuery(query="iphone 14 pro", locale="en", description="Complement trap"),
    DemoQuery(
        query="silicone mold for resin jewelry making",
        locale="en",
        description="Long-tail, niche",
    ),
    DemoQuery(
        query="unicorn holographic laptop sticker pack aesthetic",
        locale="en",
        description="Deliberate failure",
    ),
]


@router.get("/examples")
async def get_examples() -> list[DemoQuery]:
    return DEMO_QUERIES
