import api.database.orms  # noqa
import sys
import asyncio
import httpx
from api.database import get_session
from sqlalchemy import select
from api.instance.schemas import Instance
import api.miner_client as miner_client


async def get_logs(instance_id: str):
    async with get_session() as session:
        instance = (
            (await session.execute(select(Instance).where(Instance.instance_id == instance_id)))
            .unique()
            .scalar_one_or_none()
        )
        log_port = next(p for p in instance.port_mappings if p["internal_port"] == 8001)[
            "external_port"
        ]
        headers, _ = miner_client.sign_request(instance.miner_hotkey, purpose="chutes")
        client = httpx.AsyncClient(
            base_url=f"http://{instance.host}:{log_port}",
            timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
        )
        try:
            async with client.stream("GET", "/logs/stream?backfill=1000", headers=headers) as resp:
                async for chunk in resp.aiter_text():
                    cont = chunk.strip()
                    if cont not in ("", "."):
                        print(cont)
        finally:
            await client.aclose()


asyncio.run(get_logs(sys.argv[1]))
