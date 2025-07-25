"""
Routes for chutes.
"""

import re
import random
import string
import uuid
import orjson as json
import aiohttp
from loguru import logger
from slugify import slugify
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import or_, exists, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import Optional
from api.constants import EXPANSION_UTILIZATION_THRESHOLD, UNDERUTILIZED_CAP
from api.chute.schemas import (
    Chute,
    ChuteArgs,
    NodeSelector,
    ChuteUpdateArgs,
    RollingUpdate,
)
from api.chute.codecheck import is_bad_code
from api.chute.templates import (
    VLLMChuteArgs,
    VLLMEngineArgs,
    DiffusionChuteArgs,
    TEIChuteArgs,
    build_vllm_code,
    build_diffusion_code,
    build_tei_code,
)
from api.chute.response import ChuteResponse
from api.chute.util import get_chute_by_id_or_name, selector_hourly_price, get_one, is_shared
from api.user.schemas import User
from api.user.service import get_current_user, chutes_user_id
from api.image.schemas import Image
from api.image.util import get_image_by_id_or_name
from api.permissions import Permissioning

# XXX from api.instance.util import discover_chute_targets
from api.database import get_db_session, get_session
from api.pagination import PaginatedResponse
from api.fmv.fetcher import get_fetcher
from api.config import settings
from api.constants import (
    LLM_PRICE_MULT_PER_MILLION,
    DIFFUSION_PRICE_MULT_PER_STEP,
)
from api.util import ensure_is_developer, limit_deployments
from api.guesser import guesser
from api.graval_worker import handle_rolling_update

router = APIRouter()


async def _inject_current_estimated_price(chute: Chute, response: ChuteResponse):
    """
    Inject the current estimated price data into a response.
    """
    if chute.standard_template == "vllm":
        hourly = await selector_hourly_price(chute.node_selector)
        per_million = hourly * LLM_PRICE_MULT_PER_MILLION
        if chute.discount:
            per_million -= per_million * chute.discount
        response.current_estimated_price = {"per_million_tokens": {"usd": per_million}}
        tao_usd = await get_fetcher().get_price("tao")
        if tao_usd:
            response.current_estimated_price["per_million_tokens"]["tao"] = per_million / tao_usd
    elif chute.standard_template == "diffusion":
        hourly = await selector_hourly_price(chute.node_selector)
        per_step = hourly * DIFFUSION_PRICE_MULT_PER_STEP
        if chute.discount:
            per_step -= per_step * chute.discount
        response.current_estimated_price = {"per_step": {"usd": per_step}}
        tao_usd = await get_fetcher().get_price("tao")
        if tao_usd:
            response.current_estimated_price["per_step"]["tao"] = per_step / tao_usd

    # Legacy/fallback.
    if not response.current_estimated_price:
        response.current_estimated_price = {}
    node_selector = NodeSelector(**chute.node_selector)
    response.current_estimated_price.update(await node_selector.current_estimated_price())
    if chute.discount and response.current_estimated_price:
        for key in ("usd", "tao"):
            values = response.current_estimated_price.get(key)
            if values:
                for unit in values:
                    values[unit] -= values[unit] * chute.discount

    # Fix node selector return value.
    response.node_selector.update(
        {
            "compute_multiplier": node_selector.compute_multiplier,
            "supported_gpus": node_selector.supported_gpus,
        }
    )


@router.get("/", response_model=PaginatedResponse)
async def list_chutes(
    include_public: Optional[bool] = False,
    template: Optional[str] = None,
    name: Optional[str] = None,
    image: Optional[str] = None,
    slug: Optional[str] = None,
    page: Optional[int] = 0,
    limit: Optional[int] = 25,
    include_schemas: Optional[bool] = False,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes", raise_not_found=False)),
):
    """
    List (and optionally filter/paginate) chutes.
    """
    cache_key = str(
        uuid.uuid5(
            uuid.NAMESPACE_OID,
            ":".join(
                [
                    "chutes_list",
                    f"template:{template}",
                    f"image:{image}",
                    f"slug:{slug}",
                    f"page:{page}",
                    f"limit:{limit}",
                    f"name:{name}",
                    f"include_public:{include_public}",
                    f"include_schemas:{include_schemas}",
                    f"user:{current_user.user_id if current_user else None}",
                ]
            ),
        )
    ).encode()
    cached = await settings.memcache.get(cache_key)
    if cached:
        return json.loads(cached)
    query = select(Chute).options(selectinload(Chute.instances))

    # Filter by public and/or only the user's chutes.
    if current_user:
        if include_public:
            query = query.where(
                or_(
                    Chute.public.is_(True),
                    Chute.user_id == current_user.user_id,
                )
            )
        else:
            query = query.where(Chute.user_id == current_user.user_id)
    else:
        query = query.where(Chute.public.is_(True))

    # Filter by name/tag/etc.
    if name and name.strip():
        query = query.where(Chute.name.ilike(f"%{name}%"))
    if image and image.strip():
        query = query.where(
            or_(
                Image.name.ilike("%{image}%"),
                Image.tag.ilike("%{image}%"),
            )
        )
    if slug and slug.strip():
        query = query.where(Chute.slug.ilike(slug))

    # Standard template filtering.
    if template and template.strip() and template != "other":
        query = query.where(Chute.standard_template == template)
    elif template == "other":
        query = query.where(Chute.standard_template.is_(None))

    # Perform a count.
    total_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    # Pagination.
    query = (
        query.order_by(Chute.invocation_count.desc())
        .offset((page or 0) * (limit or 25))
        .limit((limit or 25))
    )

    result = await db.execute(query)
    responses = []
    cord_refs = {}
    for item in result.unique().scalars().all():
        chute_response = ChuteResponse.from_orm(item)
        cord_defs = json.dumps(item.cords).decode()
        if item.standard_template == "vllm":
            cord_defs = cord_defs.replace(f'"default":"{item.name}"', '"default":""')
        cord_ref_id = str(uuid.uuid5(uuid.NAMESPACE_OID, cord_defs))
        if cord_ref_id not in cord_refs:
            cord_refs[cord_ref_id] = item.cords
            if not include_schemas:
                for cord in cord_refs[cord_ref_id] or []:
                    cord.pop("input_schema", None)
                    cord.pop("minimal_input_schema", None)
                    cord.pop("output_schema", None)
        chute_response.cords = None
        chute_response.cord_ref_id = cord_ref_id
        responses.append(chute_response)
        await _inject_current_estimated_price(item, responses[-1])
    result = {
        "total": total,
        "page": page,
        "limit": limit,
        "items": [item.model_dump() for item in responses],
        "cord_refs": cord_refs,
    }
    await settings.memcache.set(cache_key, json.dumps(result), exptime=60)
    return result


@router.get("/rolling_updates")
async def list_rolling_updates():
    async with get_session() as session:
        result = await session.execute(text("SELECT * FROM rolling_updates"))
        columns = result.keys()
        rows = result.fetchall()
        return [dict(zip(columns, row)) for row in rows]


@router.get("/gpu_count_history")
async def get_gpu_count_history():
    query = """
        SELECT DISTINCT ON (chute_id)
            chute_id,
            (node_selector->>'gpu_count')::integer AS gpu_count
        FROM chute_history
        WHERE
            node_selector ? 'gpu_count'
            AND jsonb_typeof(node_selector->'gpu_count') = 'number'
        ORDER BY
            chute_id, created_at DESC
    """
    async with get_session(readonly=True) as session:
        results = (await session.execute(text(query))).unique().all()
        return [dict(zip(["chute_id", "gpu_count"], row)) for row in results]


@router.get("/miner_means")
async def get_chute_miner_mean_index(db: AsyncSession = Depends(get_db_session)):
    query = """
        SELECT c.chute_id, c.name
        FROM chutes c
        WHERE c.standard_template = 'vllm'
        ORDER BY invocation_count DESC
    """
    result = await db.execute(text(query))
    chutes = result.fetchall()
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chute LLM outlier index</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            ul { list-style-type: none; padding: 0; }
            li { margin: 10px 0; }
            a { text-decoration: none; color: #0066cc; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Metrics</h1>
        <ul>
    """
    for chute in chutes:
        link = f"https://api.{settings.base_domain}/chutes/miner_means/{chute.chute_id}"
        html_content += f'        <li><a href="{link}">{chute.name}</a></li>\n'
    html_content += """
        </ul>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/miner_means/{chute_id}")
@router.get("/miner_means/{chute_id}.{ext}")
async def get_chute_miner_means(
    chute_id: str,
    ext: Optional[str] = None,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Load a chute's mean TPS and output token count by miner ID.
    """
    query = """
        SELECT
            miner_hotkey,
            instance_id,
            avg_tps,
            avg_output_tokens
        FROM llm_means
        WHERE chute_id = :chute_id
        ORDER BY avg_output_tokens DESC
    """
    result = await db.execute(text(query), {"chute_id": chute_id})
    rows = result.fetchall()

    # JSON response.
    if ext == "json":
        miner_means = [
            {
                "miner_hotkey": row.miner_hotkey,
                "instance_id": row.instance_id,
                "avg_tps": float(row.avg_tps),
                "avg_output_tokens": float(row.avg_output_tokens),
            }
            for row in rows
        ]
        return JSONResponse(content=miner_means)

    # CSV response.
    if ext == "csv":
        csv_content = "instance_id,miner_hotkey,avg_tps,avg_output_tokens\n"
        for row in rows:
            csv_content += (
                f"{row.instance_id},{row.miner_hotkey},{row.avg_tps},{row.avg_output_tokens}\n"
            )
        return Response(content=csv_content, media_type="text/csv")

    # Default return an ugly hacky HTML page to make it easier to read.
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chute metrics</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #ddd; }
            .number { text-align: right; }
        </style>
    </head>
    <body>
        <h1>Metrics</h1>
        <table>
            <thead>
                <tr>
                    <th>Hotkey</th>
                    <th>Instance ID</th>
                    <th class="number">Avg TPS</th>
                    <th class="number">Avg Output Tokens</th>
                </tr>
            </thead>
            <tbody>
    """
    for row in rows:
        html_content += f"""
                <tr>
                    <td>{row.miner_hotkey}</td>
                    <td>{row.instance_id}</td>
                    <td class="number">{row.avg_tps:.2f}</td>
                    <td class="number">{row.avg_output_tokens:.2f}</td>
                </tr>
        """
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/code/{chute_id}")
async def get_chute_code(
    chute_id: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes", raise_not_found=False)),
):
    """
    Load a chute's code by ID or name.
    """
    chute = await get_one(chute_id)
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    authorized = False
    if chute.public or (
        current_user
        and (
            current_user.user_id == chute.user_id or await is_shared(chute_id, current_user.user_id)
        )
    ):
        authorized = True
    if not authorized:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    return Response(content=chute.code, media_type="text/plain")


@router.get("/utilization")
async def get_chute_utilization():
    """
    Get chute utilization data.
    """
    async with get_session(readonly=True) as session:
        query = text("""
            WITH chute_details AS (
              SELECT
                chute_id,
                (SELECT COUNT(*) FROM instances WHERE instances.chute_id = chutes.chute_id) AS live_instance_count,
                EXISTS(SELECT FROM rolling_updates WHERE chute_id = chutes.chute_id) AS update_in_progress
              FROM chutes
            )
            SELECT * FROM chute_utilization
            JOIN chute_details
            ON chute_details.chute_id = chute_utilization.chute_id;
        """)
        results = await session.execute(query)
        rows = results.mappings().all()
        utilization_data = [dict(row) for row in rows]
        for item in utilization_data:
            item["instance_count"] = item.pop("live_instance_count")
            if (
                item["avg_busy_ratio"] < EXPANSION_UTILIZATION_THRESHOLD
                and not item["total_rate_limit_errors"]
                and item["instance_count"] >= UNDERUTILIZED_CAP
            ):
                item["scalable"] = False
            else:
                item["scalable"] = True
        return utilization_data


@router.get("/{chute_id_or_name:path}", response_model=ChuteResponse)
async def get_chute(
    chute_id_or_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes", raise_not_found=False)),
):
    """
    Load a chute by ID or name.
    """
    chute = await get_chute_by_id_or_name(chute_id_or_name, db, current_user, load_instances=True)
    if not chute:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    response = ChuteResponse.from_orm(chute)
    await _inject_current_estimated_price(chute, response)
    return response


@router.post("/{chute_id}/share", response_model=ChuteResponse)
async def share_chute(
    chute_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes")),
):
    """
    Share a chute with another user.
    """
    body = await request.json()
    user_id = body.get("user_id")
    if not isinstance(user_id, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide a user_id to share to.",
        )

    # Can be username also, in which case we need to convert from username to the uuid.
    try:
        _ = uuid.UUID(user_id)
    except ValueError:
        user = (
            await db.execute(select(User).where(User.username == user_id).limit(1))
            .unique()
            .scalar_one_or_none()
        )
        user_id = user.user_id

    # Load the chute.
    chute = await get_one(chute_id)
    if not chute or chute.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )

    # Insert the share record.
    await db.execute(
        text(
            """
            INSERT INTO chute_shares
              (chute_id, shared_by, shared_to, shared_at)
            VALUES (:chute_id, :shared_by, :shared_to, NOW())
            ON CONFLICT (chute_id, shared_by, shared_to)
            DO NOTHING
            """
        ),
        {"chute_id": chute_id, "shared_by": current_user.user_id, "shared_to": user_id},
    )
    await db.commit()
    return {"ok": True}


@router.delete("/{chute_id_or_name:path}")
async def delete_chute(
    chute_id_or_name: str,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user(purpose="chutes")),
):
    """
    Delete a chute by ID or name.
    """
    chute = await get_chute_by_id_or_name(chute_id_or_name, db, current_user)
    if not chute or chute.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    chute_id = chute.chute_id
    version = chute.version
    await db.delete(chute)
    await db.commit()

    await settings.redis_client.publish(
        "miner_broadcast",
        json.dumps(
            {
                "reason": "chute_deleted",
                "data": {"chute_id": chute_id, "version": version},
            }
        ).decode(),
    )

    return {"chute_id": chute_id, "deleted": True}


async def _deploy_chute(
    chute_args: ChuteArgs,
    db: AsyncSession,
    current_user: User,
):
    """
    Deploy a chute!
    """
    image = await get_image_by_id_or_name(chute_args.image, db, current_user)
    if not image:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chute image not found, or does not belong to you",
        )
    if chute_args.public and not image.public:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Chute cannot be public when image is not public!",
        )
    version = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{image.image_id}:{chute_args.code}"))
    chute = (
        (
            await db.execute(
                select(Chute)
                .where(Chute.name.ilike(chute_args.name))
                .where(Chute.user_id == current_user.user_id)
                .options(selectinload(Chute.instances))
            )
        )
        .unique()
        .scalar_one_or_none()
    )
    if chute and chute.version == version and chute.public == chute_args.public:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chute with name={chute_args.name}, {version=} and public={chute_args.public} already exists",
        )

    # Limit h200 and b200 usage.
    if not chute_args.node_selector:
        chute_args.node_selector = {"gpu_count": 1}
    if isinstance(chute_args.node_selector, dict):
        chute_args.node_selector = NodeSelector(**chute_args.node_selector)
    if current_user.user_id != await chutes_user_id() and not current_user.has_role(
        Permissioning.unlimited_dev
    ):
        if (
            chute_args.node_selector
            and chute_args.node_selector.min_vram_gb_per_gpu
            and chute_args.node_selector.min_vram_gb_per_gpu > 80
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to require > 80gb VRAM per GPU at this time.",
            )
        if not chute_args.node_selector.exclude:
            chute_args.node_selector.exclude = []
        chute_args.node_selector.exclude = list(
            set(chute_args.node_selector.exclude or [] + ["h200", "b200", "mi300x"])
        )

        if not chute_args.node_selector.supported_gpus:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No supported GPUs based on node selector!",
            )

        # Limit h/b 200 access for now.
        if not set(chute_args.node_selector.supported_gpus) - set(["b200", "h200", "mi300x"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not allowed to require h200, b200 or mi300x at this time.",
            )

    old_version = None
    if chute:
        # Create a rolling update object so we can gracefully restart/recreate.
        permitted = {}
        for inst in chute.instances:
            if inst.miner_hotkey not in permitted:
                permitted[inst.miner_hotkey] = 0
            permitted[inst.miner_hotkey] += 1
        await db.execute(
            text(
                "DELETE FROM rolling_updates WHERE chute_id = :chute_id",
            ),
            {"chute_id": chute.chute_id},
        )
        if chute.instances:
            rolling_update = RollingUpdate(
                chute_id=chute.chute_id,
                old_version=chute.version,
                new_version=version,
                permitted=permitted,
            )
            db.add(rolling_update)

        old_version = chute.version
        chute.image_id = image.image_id
        chute.tagline = chute_args.tagline
        chute.readme = chute_args.readme
        chute.code = chute_args.code
        chute.node_selector = chute_args.node_selector
        chute.tool_description = chute_args.tool_description
        chute.filename = chute_args.filename
        chute.ref_str = chute_args.ref_str
        chute.version = version
        chute.public = chute_args.public
        chute.logo_id = (
            chute_args.logo_id if chute_args.logo_id and chute_args.logo_id.strip() else None
        )
        chute.chutes_version = image.chutes_version
        chute.cords = chute_args.cords
        chute.jobs = chute_args.jobs
        chute.concurrency = chute_args.concurrency
        chute.updated_at = func.now()
        chute.boost = None
    else:
        try:
            chute = Chute(
                chute_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_OID, f"{current_user.username}::chute::{chute_args.name}"
                    )
                ),
                image_id=image.image_id,
                user_id=current_user.user_id,
                name=chute_args.name,
                tagline=chute_args.tagline,
                readme=chute_args.readme,
                tool_description=chute_args.tool_description,
                logo_id=chute_args.logo_id if chute_args.logo_id else None,
                code=chute_args.code,
                filename=chute_args.filename,
                ref_str=chute_args.ref_str,
                version=version,
                public=chute_args.public,
                cords=chute_args.cords,
                jobs=chute_args.jobs,
                node_selector=chute_args.node_selector,
                standard_template=chute_args.standard_template,
                chutes_version=image.chutes_version,
                concurrency=chute_args.concurrency,
                boost=None,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Validation failure: {exc}",
            )

        # Generate a unique slug (subdomain).
        chute.slug = re.sub(
            r"[^a-z0-9-]+$",
            "-",
            slugify(f"{current_user.username}-{chute.name}", max_length=58).lower(),
        )
        base_slug = chute.slug
        already_exists = (
            await db.execute(select(exists().where(Chute.slug == chute.slug)))
        ).scalar()
        while already_exists:
            suffix = "".join(
                random.choice(string.ascii_lowercase + string.digits) for _ in range(5)
            )
            chute.slug = f"{base_slug}-{suffix}"
            already_exists = (
                await db.execute(select(exists().where(Chute.slug == chute.slug)))
            ).scalar()

        db.add(chute)

    # Make sure we have at least one cord or one job definition.
    if not chute.cords and not chute.jobs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A chute must define at least one cord() or job() function!",
        )

    # Limit h/b 200 access for now.
    supported_gpus = set((chute.node_selector or {}).get("supported_gpus", []))
    if (
        not (supported_gpus - set(["b200", "h200", "mi300x"]))
        and chute.user_id != await chutes_user_id()
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to require h200, b200 or mi300x at this time.",
        )

    await db.commit()
    await db.refresh(chute)

    if old_version:
        await handle_rolling_update.kiq(chute.chute_id, chute.version)
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_updated",
                    "data": {
                        "chute_id": chute.chute_id,
                        "version": chute.version,
                        "job_only": not chute.cords,
                    },
                }
            ).decode(),
        )
    else:
        await settings.redis_client.publish(
            "miner_broadcast",
            json.dumps(
                {
                    "reason": "chute_created",
                    "data": {
                        "chute_id": chute.chute_id,
                        "version": chute.version,
                        "job_only": not chute.cords,
                    },
                }
            ).decode(),
        )
    return await get_chute_by_id_or_name(chute.chute_id, db, current_user, load_instances=True)


@router.post("/", response_model=ChuteResponse)
async def deploy_chute(
    chute_args: ChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Standard deploy from the CDK.
    """
    await ensure_is_developer(db, current_user)
    await limit_deployments(db, current_user)
    if current_user.user_id not in (
        await chutes_user_id(),
        "b167f56b-3e8d-5ffa-88bf-5cc6513bb6f4",
        "5bf8a979-ea71-54bf-8644-26a3411a3b58",
    ):
        bad, response = await is_bad_code(chute_args.code)
        logger.warning(
            f"CODECHECK FAIL: User {current_user.user_id} attempted to deploy bad code {response}\n{chute_args.code}"
        )
        if bad:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=json.dumps(response).decode(),
            )
    return await _deploy_chute(chute_args, db, current_user)


async def _find_latest_image(db: AsyncSession, name: str) -> Image:
    """
    Find the latest vllm/diffusion image.
    """
    chute_user = (
        await db.execute(select(User).where(User.username == "chutes"))
    ).scalar_one_or_none()
    query = (
        select(Image)
        .where(Image.name == name)
        .where(Image.user_id == chute_user.user_id)
        .where(Image.tag != "0.8.3")
        .order_by(Image.created_at.desc())
        .limit(1)
    )
    return (await db.execute(query)).scalar_one_or_none()


def chute_to_cords(chute: Chute):
    """
    Get all cords for a chute.
    """
    return [
        {
            "method": cord._method,
            "path": cord.path,
            "public_api_path": cord.public_api_path,
            "public_api_method": cord._public_api_method,
            "stream": cord._stream,
            "function": cord._func.__name__,
            "input_schema": cord.input_schema,
            "output_schema": cord.output_schema,
            "output_content_type": cord.output_content_type,
            "minimal_input_schema": cord.minimal_input_schema,
        }
        for cord in chute._cords
    ]


@router.post("/vllm", response_model=ChuteResponse)
async def easy_deploy_vllm_chute(
    args: VLLMChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Easy/templated vLLM deployment.
    """
    await ensure_is_developer(db, current_user)
    await limit_deployments(db, current_user)

    # Make sure we can download the model, set max model length.
    if not args.engine_args:
        args.engine_args = VLLMEngineArgs()
    gated_model = False
    llama_model = False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://huggingface.co/{args.model}/resolve/main/config.json"
            ) as resp:
                if resp.status == 401:
                    gated_model = True
                resp.raise_for_status()
                try:
                    config = await resp.json()
                except Exception:
                    config = json.loads(await resp.text())
                length = config.get("max_position_embeddings", config.get("model_max_length"))
                if any(
                    [
                        arch.lower() == "llamaforcausallm"
                        for arch in config.get("architectures") or []
                    ]
                ):
                    llama_model = True
                if isinstance(length, str) and length.isidigit():
                    length = int(length)
                if isinstance(length, int):
                    if length <= 16384:
                        if (
                            not args.engine_args.max_model_len
                            or args.engine_args.max_model_len > length
                        ):
                            logger.info(
                                f"Setting max_model_len to {length} due to config.json, model={args.model}"
                            )
                            args.engine_args.max_model_len = length
                    elif not args.engine_args.max_model_len:
                        logger.info(
                            f"Setting max_model_len to 16384 due to excessively large context length in config.json, model={args.model}"
                        )
                        args.engine_args.max_model_len = 16384

        # Also check the tokenizer.
        if not args.engine_args.tokenizer:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://huggingface.co/{args.model}/resolve/main/tokenizer_config.json"
                ) as resp:
                    if resp.status == 404:
                        args.engine_args.tokenizer = "unsloth/Llama-3.2-1B-Instruct"
                    resp.raise_for_status()
                    try:
                        config = await resp.json()
                    except Exception:
                        config = json.loads(await resp.text())
                    if not config.get("chat_template"):
                        if config.get("tokenizer_class") == "tokenizer_class" and llama_model:
                            args.engine_args.tokenizer = "unsloth/Llama-3.2-1B-Instruct"
                            logger.warning(
                                f"Chat template not specified in {args.model}, defaulting to llama3"
                            )
                        elif config.get("tokenizer_class") == "LlamaTokenizer":
                            args.engine_args.tokenizer = "jondurbin/bagel-7b-v0.1"
                            logger.warning(
                                f"Chat template not specified in {args.model}, defaulting to llama2 (via bagel)"
                            )
    except Exception as exc:
        logger.warning(f"Error checking model tokenizer_config.json: {exc}")

    # Reject gaited models, e.g. meta-llama/*
    if gated_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {args.model} appears to have gated access, config.json could not be downloaded",
        )

    image = await _find_latest_image(db, "vllm")
    image = f"chutes/{image.name}:{image.tag}"
    if args.engine_args.max_model_len <= 0:
        args.engine_args.max_model_len = 16384
    code, chute = build_vllm_code(args, current_user.username, image)
    if (node_selector := args.node_selector) is None:
        async with aiohttp.ClientSession() as session:
            try:
                requirements = await guesser.analyze_model(args.model, session)
                node_selector = NodeSelector(
                    gpu_count=requirements.required_gpus,
                    min_vram_gb_per_gpu=requirements.min_vram_per_gpu,
                )
            except Exception:
                node_selector = NodeSelector(gpu_count=1, min_vram_gb_per_gpu=80)
    chute_args = ChuteArgs(
        name=args.model,
        image=image,
        tagline=args.tagline,
        readme=args.readme,
        tool_description=args.tool_description,
        logo_id=args.logo_id if args.logo_id and args.logo_id.strip() else None,
        public=args.public,
        code=code,
        filename="chute.py",
        ref_str="chute:chute",
        standard_template="vllm",
        node_selector=node_selector,
        cords=chute_to_cords(chute.chute),
        jobs=[],
    )
    return await _deploy_chute(chute_args, db, current_user)


@router.post("/diffusion", response_model=ChuteResponse)
async def easy_deploy_diffusion_chute(
    args: DiffusionChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Easy/templated diffusion deployment.
    """
    await ensure_is_developer(db, current_user)
    await limit_deployments(db, current_user)

    image = await _find_latest_image(db, "diffusion")
    image = f"chutes/{image.name}:{image.tag}"
    code, chute = build_diffusion_code(args, current_user.username, image)
    if (node_selector := args.node_selector) is None:
        node_selector = NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=24,
        )
    chute_args = ChuteArgs(
        name=args.name,
        image=image,
        tagline=args.tagline,
        readme=args.readme,
        tool_description=args.tool_description,
        logo_id=args.logo_id if args.logo_id and args.logo_id.strip() else None,
        public=args.public,
        code=code,
        filename="chute.py",
        ref_str="chute:chute",
        standard_template="diffusion",
        node_selector=node_selector,
        cords=chute_to_cords(chute.chute),
        jobs=[],
    )
    return await _deploy_chute(chute_args, db, current_user)


@router.post("/tei", response_model=ChuteResponse)
async def easy_deploy_tei_chute(
    args: TEIChuteArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Easy/templated text-embeddings-inference deployment.
    """
    await ensure_is_developer(db, current_user)
    await limit_deployments(db, current_user)

    image = await _find_latest_image(db, "tei")
    image = f"chutes/{image.name}:{image.tag}"
    code, chute = build_tei_code(args, current_user.username, image)
    if (node_selector := args.node_selector) is None:
        node_selector = NodeSelector(
            gpu_count=1,
            min_vram_gb_per_gpu=16,
        )
    node_selector.exclude = list(
        set(
            node_selector.exclude
            or [] + ["h200", "b200", "h100", "h100_sxm", "h100_nvl", "h800", "mi300x"]
        )
    )

    chute_args = ChuteArgs(
        name=args.model,
        image=image,
        tagline=args.tagline,
        readme=args.readme,
        tool_description=args.tool_description,
        logo_id=args.logo_id if args.logo_id and args.logo_id.strip() else None,
        public=args.public,
        code=code,
        filename="chute.py",
        ref_str="chute:chute",
        standard_template="tei",
        node_selector=node_selector,
        cords=chute_to_cords(chute.chute),
        jobs=[],
    )
    return await _deploy_chute(chute_args, db, current_user)


@router.put("/{chute_id_or_name:path}", response_model=ChuteResponse)
async def update_common_attributes(
    chute_id_or_name: str,
    args: ChuteUpdateArgs,
    db: AsyncSession = Depends(get_db_session),
    current_user: User = Depends(get_current_user()),
):
    """
    Update readme, tagline, etc. (but not code, image, etc.).
    """
    chute = await get_chute_by_id_or_name(chute_id_or_name, db, current_user, load_instances=True)
    if not chute or chute.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chute not found, or does not belong to you",
        )
    if args.tagline and args.tagline.strip():
        chute.tagline = args.tagline
    if args.readme and args.readme.strip():
        chute.readme = args.readme
    if args.tool_description and args.tool_description.strip():
        chute.tool_description = args.tool_description
    if args.logo_id:
        chute.logo_id = args.logo_id
    await db.commit()
    await db.refresh(chute)
    return chute
