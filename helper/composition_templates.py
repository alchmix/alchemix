"""
Persistent user templates — transition banners & talking-heads layouts.
Stored in MongoDB collection: composition_templates
"""
from __future__ import annotations

from datetime import datetime

from bson.objectid import ObjectId

VALID_TYPES = frozenset({"transition_banner", "talking_heads", "intro_teaser"})


def _normalize_type(value: str) -> str | None:
    t = (value or "").strip().lower()
    return t if t in VALID_TYPES else None


def _serialize(doc: dict) -> dict:
    out = dict(doc)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    return out


def _clear_defaults(db, user_id: str, template_type: str, channel_id=None):
    q = {
        "user_id": user_id,
        "template_type": template_type,
        "is_default": True,
    }
    if channel_id:
        q["channel_id"] = str(channel_id)
    else:
        q["$or"] = [{"channel_id": None}, {"channel_id": ""}]
    db.composition_templates.update_many(q, {"$set": {"is_default": False}})


def list_templates(db, user_id: str, template_type: str | None = None, channel_id=None):
    q = {"user_id": user_id}
    t = _normalize_type(template_type) if template_type else None
    if t:
        q["template_type"] = t
    if channel_id is not None and channel_id != "":
        q["channel_id"] = str(channel_id)
    elif channel_id == "":
        q["$or"] = [{"channel_id": None}, {"channel_id": ""}]

    rows = list(
        db.composition_templates.find(
            q,
            {
                "user_id": 0,
            },
        ).sort([("is_default", -1), ("updated_at", -1)])
    )
    return [_serialize(r) for r in rows]


def get_default_template(db, user_id: str, template_type: str, channel_id=None):
    t = _normalize_type(template_type)
    if not t:
        return None

    q = {
        "user_id": user_id,
        "template_type": t,
        "is_default": True,
    }
    if channel_id:
        q["channel_id"] = str(channel_id)
    else:
        q["$or"] = [{"channel_id": None}, {"channel_id": ""}]

    doc = db.composition_templates.find_one(q, sort=[("updated_at", -1)])
    if doc:
        return _serialize(doc)

    # Fallback: latest non-default for scope
    q2 = {"user_id": user_id, "template_type": t}
    if channel_id:
        q2["channel_id"] = str(channel_id)
    else:
        q2["$or"] = [{"channel_id": None}, {"channel_id": ""}]
    doc = db.composition_templates.find_one(q2, sort=[("updated_at", -1)])
    return _serialize(doc) if doc else None


def save_template(
    db,
    user_id: str,
    template_type: str,
    name: str,
    payload: dict,
    *,
    template_id=None,
    channel_id=None,
    set_as_default: bool = False,
):
    t = _normalize_type(template_type)
    if not t:
        raise ValueError("Invalid template_type")
    if not name:
        raise ValueError("Template name is required")
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")

    now = datetime.utcnow()
    channel_key = str(channel_id) if channel_id else None

    if set_as_default and not template_id:
        q = {
            "user_id": user_id,
            "template_type": t,
            "is_default": True,
        }
        if channel_key:
            q["channel_id"] = channel_key
        else:
            q["$or"] = [{"channel_id": None}, {"channel_id": ""}]
        existing_default = db.composition_templates.find_one(q)
        if existing_default:
            template_id = str(existing_default["_id"])

    doc = {
        "user_id": user_id,
        "template_type": t,
        "name": name[:120],
        "payload": payload,
        "channel_id": channel_key,
        "updated_at": now,
    }

    if set_as_default:
        _clear_defaults(db, user_id, t, channel_key)
        doc["is_default"] = True
    else:
        doc.setdefault("is_default", False)

    if template_id:
        try:
            oid = ObjectId(template_id)
        except Exception as exc:
            raise ValueError("Invalid template_id") from exc

        existing = db.composition_templates.find_one({"_id": oid, "user_id": user_id})
        if not existing:
            raise LookupError("Template not found")

        if set_as_default:
            doc["is_default"] = True

        db.composition_templates.update_one({"_id": oid}, {"$set": doc})
        return str(oid), "updated"

    doc["created_at"] = now
    if set_as_default:
        doc["is_default"] = True
    else:
        doc["is_default"] = False

    result = db.composition_templates.insert_one(doc)
    return str(result.inserted_id), "created"


def delete_template(db, user_id: str, template_id: str) -> bool:
    try:
        oid = ObjectId(template_id)
    except Exception:
        return False

    result = db.composition_templates.delete_one({"_id": oid, "user_id": user_id})
    return result.deleted_count > 0


def migrate_legacy_talking_heads(db, user_id: str):
    """One-time merge from old talking_heads_templates collection if present."""
    try:
        legacy = list(db.talking_heads_templates.find({"user_id": user_id}))
    except Exception:
        return

    for row in legacy:
        layout = row.get("layout_config")
        if not layout:
            continue
        exists = db.composition_templates.find_one({
            "user_id": user_id,
            "template_type": "talking_heads",
            "name": row.get("name", ""),
            "payload.layout_config": layout,
        })
        if exists:
            continue
        db.composition_templates.insert_one({
            "user_id": user_id,
            "template_type": "talking_heads",
            "name": (row.get("name") or "Imported layout")[:120],
            "payload": {"layout_config": layout},
            "channel_id": None,
            "is_default": False,
            "created_at": row.get("created_at") or datetime.utcnow(),
            "updated_at": row.get("updated_at") or datetime.utcnow(),
        })
