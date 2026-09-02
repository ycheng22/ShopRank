"""Test Cloudflare API token and Pages project accessibility."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def load_env_file() -> dict[str, str]:
    env_vars = {}
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip("\"'")
    return env_vars


def api_request(url: str, token: str) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "shoprank-validator/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return resp.status, data
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode("utf-8"))
        except Exception:
            data = {"error": str(e)}
        return e.code, data
    except Exception as e:
        return 0, {"error": str(e)}


def test_cloudflare(token: str, account_id: str | None = None, project_name: str = "shoprank") -> bool:
    print("=" * 60)
    print("Testing Cloudflare Credentials")
    print("=" * 60)

    # 1. Verify token status
    print("\n1. Verifying token authentication with Cloudflare...")
    status, data = api_request("https://api.cloudflare.com/client/v4/user/tokens/verify", token)
    if status == 200 and data.get("success"):
        result = data.get("result", {})
        print(f"   [SUCCESS] Token is valid!")
        print(f"   - Token ID: {result.get('id')}")
        print(f"   - Status: {result.get('status')}")
    else:
        print(f"   [FAILED] Token verification failed (HTTP {status})")
        print(f"   - Response: {json.dumps(data, indent=2)}")
        return False

    if not account_id:
        print("\n[!] No CLOUDFLARE_ACCOUNT_ID provided to test Pages access.")
        return True

    # 2. Check Pages projects under the account
    print(f"\n2. Testing Cloudflare Pages access for Account: {account_id}...")
    pages_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects"
    status, data = api_request(pages_url, token)
    if status == 200 and data.get("success"):
        projects = [p.get("name") for p in data.get("result", [])]
        print(f"   [SUCCESS] Successfully authenticated with Pages API!")
        print(f"   - Existing Pages projects in this account: {projects}")
    else:
        print(f"   [FAILED] Unable to access Pages for account {account_id} (HTTP {status})")
        print(f"   - Response: {json.dumps(data, indent=2)}")
        print("\n   => Reason: The token likely lacks 'Account > Cloudflare Pages > Edit' permission,")
        print("      or the account ID does not match the token's scope.")
        return False

    # 3. Check if specific project exists
    print(f"\n3. Checking for project '{project_name}'...")
    project_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects/{project_name}"
    status, data = api_request(project_url, token)
    if status == 200 and data.get("success"):
        subdomain = data.get("result", {}).get("subdomain", "")
        print(f"   [SUCCESS] Project '{project_name}' exists!")
        print(f"   - Pages URL: https://{subdomain}")
    elif status == 404:
        print(f"   [NOTE] Project '{project_name}' does not exist yet.")
        print(f"   - (The GitHub action or wrangler can create it on first deploy if the token has Edit permissions).")
    else:
        print(f"   [FAILED] Error checking project '{project_name}' (HTTP {status})")
        print(f"   - Response: {json.dumps(data, indent=2)}")
        return False

    print("\n" + "=" * 60)
    print("ALL CLOUDFLARE CHECKS PASSED!")
    print("=" * 60)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Cloudflare token and Pages setup.")
    parser.add_argument("--token", help="Cloudflare API Token")
    parser.add_argument("--account-id", help="Cloudflare Account ID")
    parser.add_argument("--project", default="shoprank", help="Project name (default: shoprank)")
    args = parser.parse_args()

    env_vars = load_env_file()
    token = args.token or os.environ.get("CLOUDFLARE_API_TOKEN") or env_vars.get("CLOUDFLARE_API_TOKEN")
    account_id = args.account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID") or env_vars.get("CLOUDFLARE_ACCOUNT_ID")

    if not token:
        print("Error: No Cloudflare API Token found.")
        print("Please provide it via --token, or set CLOUDFLARE_API_TOKEN in your .env file or environment.")
        sys.exit(1)

    ok = test_cloudflare(token, account_id, args.project)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
