import base64
import csv
import json
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

try:
    from pypinyin import Style, lazy_pinyin
except Exception:  # pragma: no cover - depends on optional runtime package availability
    class Style:  # type: ignore[no-redef]
        NORMAL = "normal"

    def lazy_pinyin(value: Any, *args: Any, **kwargs: Any) -> list[str]:  # type: ignore[no-redef]
        text = str(value or "")
        # Fallback without pypinyin: keep ASCII segments for deterministic sorting,
        # and place CJK/emoji under '#'. This keeps contacts API usable in minimal envs.
        return [text] if text.isascii() else []

from ..chat_helpers import (
    _build_avatar_url,
    _pick_avatar_url,
    _pick_display_name,
    _resolve_account_dir,
    _should_keep_session,
)
from ..path_fix import PathFixRoute
from ..wcdb_realtime import (
    WCDB_REALTIME,
    WCDBRealtimeError,
    exec_query as _wcdb_exec_query,
    get_avatar_urls as _wcdb_get_avatar_urls,
    get_contact as _wcdb_get_contact,
    get_display_names as _wcdb_get_display_names,
    get_sessions as _wcdb_get_sessions,
)

router = APIRouter(route_class=PathFixRoute)


_SYSTEM_USERNAMES = {
    "filehelper",
    "fmessage",
    "floatbottle",
    "medianote",
    "newsapp",
    "qmessage",
    "qqmail",
    "tmessage",
    "brandsessionholder",
    "brandservicesessionholder",
    "notifymessage",
    "opencustomerservicemsg",
    "notification_messages",
    "userexperience_alarm",
}

_SOURCE_SCENE_LABELS = {
    1: "通过QQ号添加",
    3: "通过微信号添加",
    6: "通过手机号添加",
    10: "通过名片添加",
    14: "通过群聊添加",
    30: "通过扫一扫添加",
}

_COUNTRY_LABELS = {
    "CN": "中国大陆",
}


class ContactTypeFilter(BaseModel):
    friends: bool = True
    groups: bool = True
    officials: bool = True


class ContactExportRequest(BaseModel):
    account: Optional[str] = Field(None, description="账号目录名（可选，默认使用第一个）")
    source: Optional[str] = Field("auto", description="数据源：auto/realtime 直接读取 WCDB；decrypted 使用旧本地库")
    output_dir: str = Field(..., description="导出目录绝对路径")
    format: str = Field("json", description="导出格式，仅支持 json/csv")
    include_avatar_link: bool = Field(True, description="是否导出 avatarLink 字段")
    contact_types: ContactTypeFilter = Field(default_factory=ContactTypeFilter)
    keyword: Optional[str] = Field(None, description="关键词筛选（可选）")


def _normalize_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _to_int(v: Any) -> int:
    try:
        return int(v or 0)
    except Exception:
        return 0


def _to_optional_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    s = _normalize_text(v)
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _normalize_contacts_source(value: Optional[str]) -> str:
    v = str(value or "").strip().lower()
    if not v:
        return "auto"
    if v in {"auto", "default", "wechat"}:
        return "auto"
    if v in {"realtime", "real-time", "wcdb"}:
        return "realtime"
    if v in {"decrypted", "local", "sqlite", "snapshot", "output"}:
        return "decrypted"
    raise HTTPException(status_code=400, detail="Invalid source, use 'auto', 'realtime' or 'decrypted'.")


def _resolve_contacts_source_for_account(source_norm: str, account_dir: Path) -> str:
    # 新模式：auto 一律走 direct WCDB。旧本地 contact.db/session.db 只在显式 decrypted 时使用。
    if source_norm == "auto":
        return "realtime"
    return source_norm


def _pick_case_insensitive_value(item: dict[str, Any], *keys: str) -> Any:
    if not isinstance(item, dict):
        return None
    for key in keys:
        if key in item:
            return item.get(key)
    lowered = {str(k).lower(): v for k, v in item.items()}
    for key in keys:
        lk = str(key).lower()
        if lk in lowered:
            return lowered.get(lk)
    return None


_PINYIN_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_PINYIN_ALPHA_RE = re.compile(r"[A-Za-z]")

# 多音字姓氏：pypinyin 对单字默认读音不一定是姓氏读音（例如：曾= ceng / zeng）。
# 这里在“姓名首字”场景优先采用常见姓氏读音，用于联系人列表的分组/排序。
_SURNAME_PINYIN_OVERRIDES: dict[str, str] = {
    "曾": "zeng",
    "区": "ou",
    "仇": "qiu",
    "解": "xie",
    "单": "shan",
    "查": "zha",
    "乐": "yue",
    "朴": "piao",
    "盖": "ge",
    "缪": "miao",
}


@lru_cache(maxsize=4096)
def _build_contact_pinyin_key(name: str) -> str:
    text = _normalize_text(name)
    if not text:
        return ""

    # Keep non-CJK segments so English names can be sorted/grouped as expected.
    first = text[0]
    override = _SURNAME_PINYIN_OVERRIDES.get(first)
    if override:
        rest = text[1:]
        parts = [override]
        if rest:
            parts.extend(lazy_pinyin(rest, style=Style.NORMAL, errors="default"))
    else:
        parts = lazy_pinyin(text, style=Style.NORMAL, errors="default")
    out: list[str] = []
    for part in parts:
        cleaned = _PINYIN_CLEAN_RE.sub("", _normalize_text(part).lower())
        if cleaned:
            out.append(cleaned)
    return "".join(out)


@lru_cache(maxsize=4096)
def _build_contact_pinyin_initial(name: str) -> str:
    text = _normalize_text(name).lstrip()
    if not text:
        return "#"

    first = text[0]
    if "A" <= first <= "Z":
        return first
    if "a" <= first <= "z":
        return first.upper()

    override = _SURNAME_PINYIN_OVERRIDES.get(first)
    if override:
        return override[0].upper()

    # For CJK, try to convert the first character to pinyin initial.
    parts = lazy_pinyin(first, style=Style.NORMAL, errors="ignore")
    if parts:
        m = _PINYIN_ALPHA_RE.search(parts[0])
        if m:
            return m.group(0).upper()

    # Emoji / digits / symbols, etc.
    return "#"


def _decode_varint(raw: bytes, offset: int) -> tuple[Optional[int], int]:
    value = 0
    shift = 0
    pos = int(offset)
    n = len(raw)
    while pos < n:
        byte = raw[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, pos
        shift += 7
        if shift > 63:
            return None, n
    return None, n


def _decode_proto_text(raw: bytes) -> str:
    if not raw:
        return ""
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()


def _coerce_blob_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, list):
        try:
            return bytes([int(x) & 0xFF for x in value])
        except Exception:
            return b""
    if isinstance(value, dict):
        for key in ("hex", "data_hex", "dataHex"):
            raw = value.get(key)
            if raw:
                return _coerce_blob_bytes(str(raw))
        for key in ("base64", "b64", "data_b64", "dataB64"):
            raw = _normalize_text(value.get(key))
            if raw:
                try:
                    return base64.b64decode(raw, validate=True)
                except Exception:
                    return b""
        return b""

    text = _normalize_text(value)
    if not text:
        return b""
    compact = re.sub(r"\s+", "", text)
    if compact.lower().startswith("0x"):
        compact = compact[2:]
    if len(compact) >= 2 and len(compact) % 2 == 0 and re.fullmatch(r"[0-9a-fA-F]+", compact):
        try:
            return bytes.fromhex(compact)
        except Exception:
            return b""
    try:
        return base64.b64decode(text, validate=True)
    except Exception:
        return b""


def _parse_contact_extra_buffer(extra_buffer: Any) -> dict[str, Any]:
    out = {
        "gender": 0,
        "signature": "",
        "country": "",
        "province": "",
        "city": "",
        "source_scene": None,
    }
    if extra_buffer is None:
        return out

    raw = _coerce_blob_bytes(extra_buffer)
    if not raw:
        return out

    idx = 0
    n = len(raw)
    while idx < n:
        tag, idx_next = _decode_varint(raw, idx)
        if tag is None:
            break
        idx = idx_next
        field_no = tag >> 3
        wire_type = tag & 0x7

        if wire_type == 0:
            val, idx_next = _decode_varint(raw, idx)
            if val is None:
                break
            idx = idx_next
            if field_no == 2:
                # 性别: 1=男, 2=女, 0=未知
                out["gender"] = int(val)
            if field_no == 8:
                out["source_scene"] = int(val)
            continue

        if wire_type == 2:
            size, idx_next = _decode_varint(raw, idx)
            if size is None:
                break
            idx = idx_next
            end = idx + int(size)
            if end > n:
                break
            chunk = raw[idx:end]
            idx = end

            if field_no in {4, 5, 6, 7}:
                text = _decode_proto_text(chunk)
                if field_no == 4:
                    out["signature"] = text
                elif field_no == 5:
                    out["country"] = text
                elif field_no == 6:
                    out["province"] = text
                elif field_no == 7:
                    out["city"] = text
            continue

        if wire_type == 1:
            idx += 8
            continue
        if wire_type == 5:
            idx += 4
            continue

        break

    return out


def _country_label(country: str) -> str:
    c = _normalize_text(country)
    if not c:
        return ""
    return _COUNTRY_LABELS.get(c.upper(), c)


def _source_scene_label(source_scene: Optional[int]) -> str:
    if source_scene is None:
        return ""
    if source_scene in _SOURCE_SCENE_LABELS:
        return _SOURCE_SCENE_LABELS[source_scene]
    return f"场景码 {source_scene}"


def _build_region(country: str, province: str, city: str) -> str:
    parts: list[str] = []
    country_text = _country_label(country)
    province_text = _normalize_text(province)
    city_text = _normalize_text(city)
    if country_text:
        parts.append(country_text)
    if province_text:
        parts.append(province_text)
    if city_text:
        parts.append(city_text)
    return "·".join(parts)


def _safe_export_part(s: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(s or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "account"


def _is_valid_contact_username(username: str) -> bool:
    u = _normalize_text(username)
    if not u:
        return False
    if u in _SYSTEM_USERNAMES:
        return False
    if u.startswith("fake_"):
        return False
    if not _should_keep_session(u, include_official=True) and not u.startswith("gh_") and u != "weixin":
        return False
    return True


def _get_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except Exception:
        return set()

    out: set[str] = set()
    for row in rows:
        try:
            name = _normalize_text(row["name"] if "name" in row.keys() else row[1]).lower()
        except Exception:
            continue
        if name:
            out.add(name)
    return out


def _build_contact_select_sql(table: str, columns: set[str]) -> Optional[str]:
    if "username" not in columns:
        return None

    specs: list[tuple[str, str, str]] = [
        ("username", "username", "''"),
        ("remark", "remark", "''"),
        ("nick_name", "nick_name", "''"),
        ("alias", "alias", "''"),
        ("local_type", "local_type", "0"),
        ("verify_flag", "verify_flag", "0"),
        ("big_head_url", "big_head_url", "''"),
        ("small_head_url", "small_head_url", "''"),
        ("description", "description", "''"),
        ("extra_buffer", "extra_buffer", "x''"),
    ]

    select_parts: list[str] = []
    for key, alias, fallback in specs:
        if key in columns:
            select_parts.append(key)
        else:
            select_parts.append(f"{fallback} AS {alias}")
    return f"SELECT {', '.join(select_parts)} FROM {table}"


def _load_contact_rows_map(contact_db_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not contact_db_path.exists():
        return out

    conn = sqlite3.connect(str(contact_db_path))
    conn.row_factory = sqlite3.Row
    try:
        def read_rows(table: str) -> list[sqlite3.Row]:
            columns = _get_table_columns(conn, table)
            sql = _build_contact_select_sql(table, columns)
            if not sql:
                return []
            try:
                return conn.execute(sql).fetchall()
            except Exception:
                return []
            return []

        for table in ("contact", "stranger"):
            rows = read_rows(table)
            for row in rows:
                username = _normalize_text(row["username"] if "username" in row.keys() else "")
                if (not username) or (username in out):
                    continue

                extra_info = _parse_contact_extra_buffer(
                    row["extra_buffer"] if "extra_buffer" in row.keys() else b""
                )
                out[username] = {
                    "username": username,
                    "remark": _normalize_text(row["remark"] if "remark" in row.keys() else ""),
                    "nick_name": _normalize_text(row["nick_name"] if "nick_name" in row.keys() else ""),
                    "alias": _normalize_text(row["alias"] if "alias" in row.keys() else ""),
                    "local_type": _to_int(row["local_type"] if "local_type" in row.keys() else 0),
                    "verify_flag": _to_int(row["verify_flag"] if "verify_flag" in row.keys() else 0),
                    "big_head_url": _normalize_text(row["big_head_url"] if "big_head_url" in row.keys() else ""),
                    "small_head_url": _normalize_text(row["small_head_url"] if "small_head_url" in row.keys() else ""),
                    "description": _normalize_text(row["description"] if "description" in row.keys() else ""),
                    "gender": _to_int(extra_info.get("gender")),
                    "signature": _normalize_text(extra_info.get("signature")),
                    "country": _normalize_text(extra_info.get("country")),
                    "province": _normalize_text(extra_info.get("province")),
                    "city": _normalize_text(extra_info.get("city")),
                    "source_scene": _to_optional_int(extra_info.get("source_scene")),
                }
        return out
    finally:
        conn.close()


def _load_session_sort_timestamps(session_db_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    if not session_db_path.exists():
        return out

    conn = sqlite3.connect(str(session_db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows: list[sqlite3.Row] = []
        queries = [
            "SELECT username, COALESCE(sort_timestamp, 0) AS ts FROM SessionTable",
            "SELECT username, COALESCE(last_timestamp, 0) AS ts FROM SessionTable",
        ]
        for sql in queries:
            try:
                rows = conn.execute(sql).fetchall()
                break
            except Exception:
                continue

        for row in rows:
            username = _normalize_text(row["username"] if "username" in row.keys() else "")
            if not username:
                continue
            ts = _to_int(row["ts"] if "ts" in row.keys() else 0)
            prev = out.get(username, 0)
            if ts > prev:
                out[username] = ts
        return out
    finally:
        conn.close()


def _load_session_group_usernames(session_db_path: Path) -> set[str]:
    out: set[str] = set()
    if not session_db_path.exists():
        return out

    conn = sqlite3.connect(str(session_db_path))
    conn.row_factory = sqlite3.Row
    try:
        queries = [
            "SELECT username FROM SessionTable",
            "SELECT username FROM sessiontable",
        ]
        for sql in queries:
            try:
                rows = conn.execute(sql).fetchall()
            except Exception:
                continue
            for row in rows:
                username = _normalize_text(row["username"] if "username" in row.keys() else "")
                if username and ("@chatroom" in username):
                    out.add(username)
            return out
        return out
    finally:
        conn.close()


def _infer_contact_type(username: str, row: dict[str, Any]) -> Optional[str]:
    if not username:
        return None

    if "@chatroom" in username:
        return "group"

    verify_flag = _to_int(row.get("verify_flag"))
    if username.startswith("gh_") or verify_flag != 0:
        return "official"

    local_type = _to_int(row.get("local_type"))
    if local_type == 1:
        return "friend"

    return None


def _matches_keyword(contact: dict[str, Any], keyword: str) -> bool:
    kw = _normalize_text(keyword).lower()
    if not kw:
        return True

    fields = [
        contact.get("username", ""),
        contact.get("displayName", ""),
        contact.get("remark", ""),
        contact.get("nickname", ""),
        contact.get("alias", ""),
        contact.get("region", ""),
        contact.get("source", ""),
        contact.get("country", ""),
        contact.get("province", ""),
        contact.get("city", ""),
    ]
    for field in fields:
        if kw in _normalize_text(field).lower():
            return True
    return False


def _contact_item_from_session(
    *,
    account_dir: Path,
    base_url: str,
    username: str,
    display_name: str,
    avatar_link: str,
    sort_ts: int,
) -> dict[str, Any]:
    contact_type = "group" if "@chatroom" in username else ("official" if username.startswith("gh_") or username == "weixin" else "friend")
    display_name = _normalize_text(display_name) or username
    return {
        "username": username,
        "displayName": display_name,
        "remark": "",
        "nickname": display_name if display_name != username else "",
        "alias": "",
        "gender": 0,
        "signature": "",
        "type": contact_type,
        "country": "",
        "province": "",
        "city": "",
        "region": "",
        "sourceScene": None,
        "source": "",
        "avatar": base_url + _build_avatar_url(account_dir.name, username),
        "avatarLink": _normalize_text(avatar_link),
        "_sortTs": int(sort_ts or 0),
    }


def _profile_contact_type(username: str, row: Optional[dict[str, Any]] = None) -> str:
    inferred = _infer_contact_type(username, row or {})
    if inferred:
        return inferred
    if "@chatroom" in username:
        return "group"
    if username.startswith("gh_") or username == "weixin":
        return "official"
    return "friend"


def _pick_contact_row_value(row: dict[str, Any], *keys: str) -> Any:
    return _pick_case_insensitive_value(row, *keys)


def _pick_contact_row_text(row: dict[str, Any], *keys: str) -> str:
    return _normalize_text(_pick_contact_row_value(row, *keys))


def _normalize_contact_profile_row(row: dict[str, Any], fallback_username: str) -> dict[str, Any]:
    username = (
        _pick_contact_row_text(row, "username", "user_name", "userName", "UserName")
        or _normalize_text(fallback_username)
    )
    extra_info = _parse_contact_extra_buffer(
        _pick_contact_row_value(row, "extra_buffer", "extraBuffer", "ExtraBuffer")
    )

    country = _pick_contact_row_text(row, "country", "Country") or _normalize_text(extra_info.get("country"))
    province = _pick_contact_row_text(row, "province", "Province") or _normalize_text(extra_info.get("province"))
    city = _pick_contact_row_text(row, "city", "City") or _normalize_text(extra_info.get("city"))
    signature = (
        _pick_contact_row_text(row, "signature", "sign", "description", "desc", "Description")
        or _normalize_text(extra_info.get("signature"))
    )
    source_scene = _to_optional_int(
        _pick_contact_row_value(row, "source_scene", "sourceScene", "SourceScene")
    )
    if source_scene is None:
        source_scene = _to_optional_int(extra_info.get("source_scene"))

    return {
        "username": username,
        "remark": _pick_contact_row_text(row, "remark", "Remark"),
        "nick_name": _pick_contact_row_text(row, "nick_name", "nickname", "nickName", "NickName"),
        "alias": _pick_contact_row_text(row, "alias", "Alias"),
        "local_type": _to_int(_pick_contact_row_value(row, "local_type", "localType", "LocalType")),
        "verify_flag": _to_int(_pick_contact_row_value(row, "verify_flag", "verifyFlag", "VerifyFlag")),
        "big_head_url": _pick_contact_row_text(
            row,
            "big_head_url",
            "bigHeadUrl",
            "bigHeadURL",
            "big_head_img_url",
            "bigHeadImgUrl",
            "avatarUrl",
        ),
        "small_head_url": _pick_contact_row_text(
            row,
            "small_head_url",
            "smallHeadUrl",
            "smallHeadURL",
            "small_head_img_url",
            "smallHeadImgUrl",
        ),
        "gender": _to_int(_pick_contact_row_value(row, "gender", "Gender") or extra_info.get("gender")),
        "signature": signature,
        "country": country,
        "province": province,
        "city": city,
        "source_scene": source_scene,
    }


def _contact_item_from_profile_row(
    *,
    account_dir: Path,
    base_url: str,
    username: str,
    row: Optional[dict[str, Any]],
    display_name_fallback: str = "",
    avatar_link_fallback: str = "",
) -> dict[str, Any]:
    normalized = _normalize_contact_profile_row(row or {}, username) if row else {"username": username}
    resolved_username = _normalize_text(normalized.get("username")) or username
    display_name = _pick_display_name(normalized, resolved_username)
    if display_name == resolved_username and display_name_fallback:
        display_name = display_name_fallback
    avatar_link = _normalize_text(_pick_avatar_url(normalized) or avatar_link_fallback)
    country = _normalize_text(normalized.get("country"))
    province = _normalize_text(normalized.get("province"))
    city = _normalize_text(normalized.get("city"))
    source_scene = _to_optional_int(normalized.get("source_scene"))
    return {
        "username": resolved_username,
        "displayName": _normalize_text(display_name) or display_name_fallback or resolved_username,
        "remark": _normalize_text(normalized.get("remark")),
        "nickname": _normalize_text(normalized.get("nick_name")),
        "alias": _normalize_text(normalized.get("alias")),
        "gender": _to_int(normalized.get("gender")),
        "signature": _normalize_text(normalized.get("signature")),
        "type": _profile_contact_type(resolved_username, normalized),
        "country": country,
        "province": province,
        "city": city,
        "region": _build_region(country, province, city),
        "sourceScene": source_scene,
        "source": _source_scene_label(source_scene),
        "avatar": base_url + _build_avatar_url(account_dir.name, resolved_username),
        "avatarLink": avatar_link,
        "pinyinKey": _build_contact_pinyin_key(_normalize_text(display_name) or resolved_username),
        "pinyinInitial": _build_contact_pinyin_initial(_normalize_text(display_name) or resolved_username),
    }


def _sql_literal(value: str) -> str:
    return "'" + str(value or "").replace("'", "''") + "'"


def _query_realtime_contact_row(handle: int, username: str) -> Optional[dict[str, Any]]:
    u = _normalize_text(username)
    if not u:
        return None

    try:
        row = _wcdb_get_contact(handle, u)
        if isinstance(row, dict) and row:
            return row
    except Exception:
        pass

    quoted = _sql_literal(u)
    for table in ("contact", "stranger"):
        try:
            rows = _wcdb_exec_query(
                handle,
                kind="contact",
                path=None,
                sql=f"SELECT * FROM {table} WHERE username={quoted} LIMIT 1",
            )
        except Exception:
            continue
        if rows:
            first = rows[0]
            if isinstance(first, dict):
                return first
    return None


def _get_contact_profile_realtime(
    *,
    account_dir: Path,
    base_url: str,
    username: str,
) -> tuple[dict[str, Any], bool]:
    try:
        rt_conn = WCDB_REALTIME.ensure_connected(account_dir)
    except WCDBRealtimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Realtime contact profile unavailable: {e}")

    row: Optional[dict[str, Any]] = None
    display_name_fallback = ""
    avatar_link_fallback = ""
    try:
        with rt_conn.lock:
            row = _query_realtime_contact_row(rt_conn.handle, username)
            if row is None:
                try:
                    display_name_fallback = _normalize_text(_wcdb_get_display_names(rt_conn.handle, [username]).get(username))
                except Exception:
                    display_name_fallback = ""
                try:
                    avatar_link_fallback = _normalize_text(_wcdb_get_avatar_urls(rt_conn.handle, [username]).get(username))
                except Exception:
                    avatar_link_fallback = ""
    except WCDBRealtimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Realtime contact profile lookup failed: {e}")

    return (
        _contact_item_from_profile_row(
            account_dir=account_dir,
            base_url=base_url,
            username=username,
            row=row,
            display_name_fallback=display_name_fallback,
            avatar_link_fallback=avatar_link_fallback,
        ),
        row is not None,
    )


def _get_contact_profile_decrypted(
    *,
    account_dir: Path,
    base_url: str,
    username: str,
) -> tuple[dict[str, Any], bool]:
    contact_rows = _load_contact_rows_map(account_dir / "contact.db")
    row = contact_rows.get(username)
    return (
        _contact_item_from_profile_row(
            account_dir=account_dir,
            base_url=base_url,
            username=username,
            row=row,
        ),
        row is not None,
    )


def _collect_contacts_for_account_realtime(
    *,
    account_dir: Path,
    base_url: str,
    keyword: Optional[str],
    include_friends: bool,
    include_groups: bool,
    include_officials: bool,
) -> list[dict[str, Any]]:
    if not (include_friends or include_groups or include_officials):
        return []

    try:
        rt_conn = WCDB_REALTIME.ensure_connected(account_dir)
    except WCDBRealtimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Realtime contacts unavailable: {e}")

    try:
        with rt_conn.lock:
            raw_sessions = _wcdb_get_sessions(rt_conn.handle)
    except WCDBRealtimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Realtime session lookup failed: {e}")

    session_rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in raw_sessions or []:
        if not isinstance(item, dict):
            continue
        username = _normalize_text(
            _pick_case_insensitive_value(item, "username", "user_name", "UserName")
        )
        if not username or username in seen:
            continue
        seen.add(username)
        if not _is_valid_contact_username(username):
            continue
        contact_type = "group" if "@chatroom" in username else ("official" if username.startswith("gh_") or username == "weixin" else "friend")
        if contact_type == "friend" and not include_friends:
            continue
        if contact_type == "group" and not include_groups:
            continue
        if contact_type == "official" and not include_officials:
            continue
        sort_ts = _to_int(
            _pick_case_insensitive_value(item, "sort_timestamp", "sortTimestamp", "last_timestamp", "lastTimestamp")
        )
        session_rows.append((username, sort_ts))

    usernames = [u for u, _ in session_rows]
    display_names: dict[str, str] = {}
    avatar_links: dict[str, str] = {}
    if usernames:
        try:
            with rt_conn.lock:
                display_names = _wcdb_get_display_names(rt_conn.handle, usernames)
                avatar_links = _wcdb_get_avatar_urls(rt_conn.handle, usernames)
        except Exception:
            display_names = {}
            avatar_links = {}

    contacts: list[dict[str, Any]] = []
    for username, sort_ts in session_rows:
        item = _contact_item_from_session(
            account_dir=account_dir,
            base_url=base_url,
            username=username,
            display_name=str(display_names.get(username) or username),
            avatar_link=str(avatar_links.get(username) or ""),
            sort_ts=int(sort_ts or 0),
        )
        if not _matches_keyword(item, keyword or ""):
            continue
        contacts.append(item)

    contacts.sort(
        key=lambda x: (
            -_to_int(x.get("_sortTs", 0)),
            _normalize_text(x.get("displayName", "")).lower(),
            _normalize_text(x.get("username", "")).lower(),
        )
    )
    for item in contacts:
        item.pop("_sortTs", None)
        name_for_pinyin = _normalize_text(item.get("displayName")) or _normalize_text(item.get("username"))
        item["pinyinKey"] = _build_contact_pinyin_key(name_for_pinyin)
        item["pinyinInitial"] = _build_contact_pinyin_initial(name_for_pinyin)
    return contacts


def _collect_contacts_for_account(
    *,
    account_dir: Path,
    base_url: str,
    keyword: Optional[str],
    include_friends: bool,
    include_groups: bool,
    include_officials: bool,
    source: Optional[str] = None,
) -> list[dict[str, Any]]:
    if not (include_friends or include_groups or include_officials):
        return []

    source_norm = _resolve_contacts_source_for_account(_normalize_contacts_source(source), account_dir)
    if source_norm == "realtime":
        return _collect_contacts_for_account_realtime(
            account_dir=account_dir,
            base_url=base_url,
            keyword=keyword,
            include_friends=include_friends,
            include_groups=include_groups,
            include_officials=include_officials,
        )

    contact_db_path = account_dir / "contact.db"
    session_db_path = account_dir / "session.db"
    contact_rows = _load_contact_rows_map(contact_db_path)
    session_ts_map = _load_session_sort_timestamps(session_db_path)
    session_group_usernames = _load_session_group_usernames(session_db_path)

    contacts: list[dict[str, Any]] = []
    for username, row in contact_rows.items():
        if not _is_valid_contact_username(username):
            continue

        contact_type = _infer_contact_type(username, row)
        if contact_type is None:
            continue
        if contact_type == "friend" and not include_friends:
            continue
        if contact_type == "group" and not include_groups:
            continue
        if contact_type == "official" and not include_officials:
            continue

        display_name = _pick_display_name(row, username)
        if not display_name:
            display_name = username

        avatar_link = _normalize_text(_pick_avatar_url(row) or "")
        avatar = base_url + _build_avatar_url(account_dir.name, username)
        country = _normalize_text(row.get("country"))
        province = _normalize_text(row.get("province"))
        city = _normalize_text(row.get("city"))
        source_scene = _to_optional_int(row.get("source_scene"))
        gender = _to_int(row.get("gender"))
        signature = _normalize_text(row.get("signature"))

        item = {
            "username": username,
            "displayName": display_name,
            "remark": _normalize_text(row.get("remark")),
            "nickname": _normalize_text(row.get("nick_name")),
            "alias": _normalize_text(row.get("alias")),
            "gender": gender,
            "signature": signature,
            "type": contact_type,
            "country": country,
            "province": province,
            "city": city,
            "region": _build_region(country, province, city),
            "sourceScene": source_scene,
            "source": _source_scene_label(source_scene),
            "avatar": avatar,
            "avatarLink": avatar_link,
            "_sortTs": _to_int(session_ts_map.get(username, 0)),
        }

        if not _matches_keyword(item, keyword or ""):
            continue
        contacts.append(item)

    if include_groups:
        for username in session_group_usernames:
            if username in contact_rows:
                continue
            if not _is_valid_contact_username(username):
                continue

            avatar_link = ""
            avatar = base_url + _build_avatar_url(account_dir.name, username)

            item = {
                "username": username,
                "displayName": username,
                "remark": "",
                "nickname": "",
                "alias": "",
                "gender": 0,
                "signature": "",
                "type": "group",
                "country": "",
                "province": "",
                "city": "",
                "region": "",
                "sourceScene": None,
                "source": "",
                "avatar": avatar,
                "avatarLink": avatar_link,
                "_sortTs": _to_int(session_ts_map.get(username, 0)),
            }

            if not _matches_keyword(item, keyword or ""):
                continue
            contacts.append(item)

    contacts.sort(
        key=lambda x: (
            -_to_int(x.get("_sortTs", 0)),
            _normalize_text(x.get("displayName", "")).lower(),
            _normalize_text(x.get("username", "")).lower(),
        )
    )
    for item in contacts:
        item.pop("_sortTs", None)
        name_for_pinyin = _normalize_text(item.get("displayName")) or _normalize_text(item.get("username"))
        item["pinyinKey"] = _build_contact_pinyin_key(name_for_pinyin)
        item["pinyinInitial"] = _build_contact_pinyin_initial(name_for_pinyin)
    return contacts


def _build_counts(contacts: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "friends": 0,
        "groups": 0,
        "officials": 0,
        "total": 0,
    }
    for item in contacts:
        t = _normalize_text(item.get("type"))
        if t == "friend":
            counts["friends"] += 1
        elif t == "group":
            counts["groups"] += 1
        elif t == "official":
            counts["officials"] += 1
    counts["total"] = len(contacts)
    return counts


def _build_export_contacts(
    contacts: list[dict[str, Any]],
    *,
    include_avatar_link: bool,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in contacts:
        row = {
            "username": _normalize_text(item.get("username")),
            "displayName": _normalize_text(item.get("displayName")),
            "remark": _normalize_text(item.get("remark")),
            "nickname": _normalize_text(item.get("nickname")),
            "alias": _normalize_text(item.get("alias")),
            "type": _normalize_text(item.get("type")),
            "region": _normalize_text(item.get("region")),
            "country": _normalize_text(item.get("country")),
            "province": _normalize_text(item.get("province")),
            "city": _normalize_text(item.get("city")),
            "source": _normalize_text(item.get("source")),
            "sourceScene": _to_optional_int(item.get("sourceScene")),
        }
        if include_avatar_link:
            row["avatarLink"] = _normalize_text(item.get("avatarLink"))
        out.append(row)
    return out


def _write_json_export(
    output_path: Path,
    *,
    account: str,
    source: str,
    contacts: list[dict[str, Any]],
    include_avatar_link: bool,
    keyword: str,
    contact_types: ContactTypeFilter,
) -> None:
    payload = {
        "exportedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": account,
        "source": source,
        "count": len(contacts),
        "filters": {
            "keyword": keyword,
            "contactTypes": {
                "friends": bool(contact_types.friends),
                "groups": bool(contact_types.groups),
                "officials": bool(contact_types.officials),
            },
            "includeAvatarLink": bool(include_avatar_link),
        },
        "contacts": contacts,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv_export(
    output_path: Path,
    *,
    contacts: list[dict[str, Any]],
    include_avatar_link: bool,
) -> None:
    columns: list[tuple[str, str]] = [
        ("username", "用户名"),
        ("displayName", "显示名称"),
        ("remark", "备注"),
        ("nickname", "昵称"),
        ("alias", "微信号"),
        ("type", "类型"),
        ("region", "地区"),
        ("country", "国家/地区码"),
        ("province", "省份"),
        ("city", "城市"),
        ("source", "来源"),
        ("sourceScene", "来源场景码"),
    ]
    if include_avatar_link:
        columns.append(("avatarLink", "头像链接"))

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label for _, label in columns])
        for item in contacts:
            writer.writerow([_normalize_text(item.get(key, "")) for key, _ in columns])


@router.get("/api/chat/contacts/profile", summary="获取单个联系人资料")
def get_chat_contact_profile(
    request: Request,
    account: Optional[str] = None,
    username: str = "",
    source: Optional[str] = None,
):
    account_dir = _resolve_account_dir(account)
    source_norm = _resolve_contacts_source_for_account(_normalize_contacts_source(source), account_dir)
    base_url = str(request.base_url).rstrip("/")
    normalized_username = _normalize_text(username)
    if not normalized_username:
        raise HTTPException(status_code=400, detail="username is required.")

    if source_norm == "realtime":
        contact, found = _get_contact_profile_realtime(
            account_dir=account_dir,
            base_url=base_url,
            username=normalized_username,
        )
    else:
        contact, found = _get_contact_profile_decrypted(
            account_dir=account_dir,
            base_url=base_url,
            username=normalized_username,
        )

    return {
        "status": "success",
        "account": account_dir.name,
        "source": source_norm,
        "found": bool(found),
        "contact": contact,
    }


@router.get("/api/chat/contacts", summary="获取联系人列表")
def list_chat_contacts(
    request: Request,
    account: Optional[str] = None,
    source: Optional[str] = None,
    keyword: Optional[str] = None,
    include_friends: bool = True,
    include_groups: bool = True,
    include_officials: bool = True,
):
    account_dir = _resolve_account_dir(account)
    source_norm = _resolve_contacts_source_for_account(_normalize_contacts_source(source), account_dir)
    base_url = str(request.base_url).rstrip("/")

    contacts = _collect_contacts_for_account(
        account_dir=account_dir,
        base_url=base_url,
        keyword=keyword,
        include_friends=bool(include_friends),
        include_groups=bool(include_groups),
        include_officials=bool(include_officials),
        source=source_norm,
    )

    return {
        "status": "success",
        "account": account_dir.name,
        "source": source_norm,
        "total": len(contacts),
        "counts": _build_counts(contacts),
        "contacts": contacts,
    }


@router.post("/api/chat/contacts/export", summary="导出联系人")
def export_chat_contacts(request: Request, req: ContactExportRequest):
    account_dir = _resolve_account_dir(req.account)
    source_norm = _resolve_contacts_source_for_account(_normalize_contacts_source(req.source), account_dir)

    output_dir_raw = _normalize_text(req.output_dir)
    if not output_dir_raw:
        raise HTTPException(status_code=400, detail="output_dir is required.")

    output_dir = Path(output_dir_raw).expanduser()
    if not output_dir.is_absolute():
        raise HTTPException(status_code=400, detail="output_dir must be an absolute path.")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to prepare output_dir: {e}")

    base_url = str(request.base_url).rstrip("/")
    contacts = _collect_contacts_for_account(
        account_dir=account_dir,
        base_url=base_url,
        keyword=req.keyword,
        include_friends=bool(req.contact_types.friends),
        include_groups=bool(req.contact_types.groups),
        include_officials=bool(req.contact_types.officials),
        source=source_norm,
    )

    export_contacts = _build_export_contacts(
        contacts,
        include_avatar_link=bool(req.include_avatar_link),
    )

    fmt = _normalize_text(req.format).lower()
    if fmt not in {"json", "csv"}:
        raise HTTPException(status_code=400, detail="Unsupported format, use 'json' or 'csv'.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_account = _safe_export_part(account_dir.name)
    output_path = output_dir / f"contacts_{safe_account}_{ts}.{fmt}"

    try:
        if fmt == "json":
            _write_json_export(
                output_path,
                account=account_dir.name,
                source=source_norm,
                contacts=export_contacts,
                include_avatar_link=bool(req.include_avatar_link),
                keyword=_normalize_text(req.keyword),
                contact_types=req.contact_types,
            )
        else:
            _write_csv_export(
                output_path,
                contacts=export_contacts,
                include_avatar_link=bool(req.include_avatar_link),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export contacts: {e}")

    return {
        "status": "success",
        "account": account_dir.name,
        "source": source_norm,
        "format": fmt,
        "outputPath": str(output_path),
        "count": len(export_contacts),
    }
