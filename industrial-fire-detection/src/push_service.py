import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Graceful optional imports so app never crashes if pywebpush is missing
try:
    from py_vapid import Vapid, b64urlencode
    from cryptography.hazmat.primitives import serialization
    import pywebpush
    HAS_WEBPUSH = True
except ImportError as err:
    HAS_WEBPUSH = False
    logger.warning(f"pywebpush / py_vapid not available ({err}). Push notifications will be disabled.")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KEYS_FILE = os.path.join(DATA_DIR, "vapid_keys.json")
SUBSCRIPTIONS_FILE = os.path.join(DATA_DIR, "push_subscriptions.json")


def get_or_create_vapid_keys() -> Dict[str, str]:
    """
    Retrieves existing VAPID keys from environment or local storage,
    or generates a fresh pair and persists them.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Check environment variables
    env_pub = os.getenv("VAPID_PUBLIC_KEY")
    env_priv = os.getenv("VAPID_PRIVATE_KEY")
    env_sub = os.getenv("VAPID_CLAIMS_SUB", "mailto:admin@firealert.local")
    
    if env_pub and env_priv:
        return {
            "public_key": env_pub.strip(),
            "private_key": env_priv.strip(),
            "claims_sub": env_sub.strip()
        }
        
    # 2. Check keys file
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("public_key") and data.get("private_key"):
                    return data
        except Exception as e:
            logger.warning(f"Failed to read {KEYS_FILE}: {e}")

    # 3. Generate new VAPID keys if py_vapid is installed
    if not HAS_WEBPUSH:
        return {
            "public_key": "",
            "private_key": "",
            "claims_sub": env_sub
        }

    try:
        vapid = Vapid()
        vapid.generate_keys()
        raw_pub = vapid.public_key.public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint
        )
        b64_pub = b64urlencode(raw_pub)
        pem_priv = vapid.private_pem().decode("utf-8")
        
        keys_data = {
            "public_key": b64_pub,
            "private_key": pem_priv,
            "claims_sub": env_sub
        }
        
        with open(KEYS_FILE, "w", encoding="utf-8") as f:
            json.dump(keys_data, f, indent=2)
        logger.info(f"Generated and saved new VAPID keys to {KEYS_FILE}")
        return keys_data
    except Exception as e:
        logger.error(f"Failed to generate VAPID keys: {e}")
        return {
            "public_key": "",
            "private_key": "",
            "claims_sub": env_sub
        }


def get_public_key() -> str:
    """Returns the base64url-encoded public key for browser registration."""
    keys = get_or_create_vapid_keys()
    return keys.get("public_key", "")


def load_subscriptions() -> List[Dict[str, Any]]:
    """Loads all stored push subscriptions."""
    if not os.path.exists(SUBSCRIPTIONS_FILE):
        return []
    try:
        with open(SUBSCRIPTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading push subscriptions: {e}")
        return []


def save_subscriptions(subs: List[Dict[str, Any]]) -> None:
    """Saves push subscriptions list to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(SUBSCRIPTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(subs, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving push subscriptions: {e}")


def add_subscription(sub_info: Dict[str, Any]) -> bool:
    """
    Registers a new push subscription or updates an existing one.
    """
    endpoint = sub_info.get("endpoint")
    if not endpoint:
        return False
        
    subs = load_subscriptions()
    subs = [s for s in subs if s.get("endpoint") != endpoint]
    subs.append(sub_info)
    save_subscriptions(subs)
    return True


def remove_subscription(endpoint: str) -> bool:
    """Removes a subscription matching endpoint."""
    subs = load_subscriptions()
    new_subs = [s for s in subs if s.get("endpoint") != endpoint]
    if len(new_subs) != len(subs):
        save_subscriptions(new_subs)
        return True
    return False


def send_push_notification(title: str, body: str, url: str = "/dashboard", tag: str = "fire-alert", icon: str = "/static/firefighter-logo.png") -> Dict[str, Any]:
    """
    Broadcasts a push notification to all stored subscriptions.
    Automatically prunes expired/unregistered endpoints (HTTP 404, 410).
    """
    if not HAS_WEBPUSH:
        return {
            "total": 0,
            "sent": 0,
            "failed": 0,
            "message": "Web Push disabled: 'pywebpush' package not installed in current Python environment."
        }

    keys = get_or_create_vapid_keys()
    private_key = keys.get("private_key")
    if not private_key:
        return {"total": 0, "sent": 0, "failed": 0, "message": "VAPID keys not configured."}
        
    claims_sub = keys.get("claims_sub", "mailto:admin@firealert.local")
    
    subs = load_subscriptions()
    if not subs:
        return {"total": 0, "sent": 0, "failed": 0, "message": "No active push subscribers"}
        
    payload = json.dumps({
        "title": title,
        "body": body,
        "url": url,
        "tag": tag,
        "icon": icon,
        "badge": icon
    })
    
    sent_count = 0
    failed_count = 0
    expired_endpoints = []
    
    for sub in subs:
        try:
            pywebpush.webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=private_key,
                vapid_claims={"sub": claims_sub},
                ttl=3600
            )
            sent_count += 1
        except pywebpush.WebPushException as ex:
            failed_count += 1
            logger.warning(f"WebPush failed for subscriber: {ex}")
            if ex.response is not None and ex.response.status_code in [404, 410]:
                expired_endpoints.append(sub.get("endpoint"))
        except Exception as ex:
            failed_count += 1
            logger.warning(f"Unexpected error in WebPush: {ex}")

    if expired_endpoints:
        subs = [s for s in subs if s.get("endpoint") not in expired_endpoints]
        save_subscriptions(subs)
        
    return {
        "total": len(subs) + len(expired_endpoints),
        "sent": sent_count,
        "failed": failed_count,
        "pruned": len(expired_endpoints)
    }
