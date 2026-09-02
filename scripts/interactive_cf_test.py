import getpass
import json
import urllib.request
import urllib.error
import sys

def main():
    print("=== Cloudflare Pages Debugger ===")
    token = getpass.getpass("Paste your Cloudflare API Token (from GitHub Secrets): ")
    account_id = input("Paste your Cloudflare Account ID (from GitHub Secrets): ").strip()

    print("\n1. Verifying Token...")
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/user/tokens/verify",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            print(" -> [SUCCESS] Token is VALID and authenticated.")
    except urllib.error.HTTPError as e:
        print(f" -> [FAILED] Token is INVALID. HTTP {e.code}")
        sys.exit(1)

    print(f"\n2. Checking projects in Account {account_id}...")
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/pages/projects",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            projects = [p['name'] for p in data.get('result', [])]
            print(f" -> [SUCCESS] Found projects: {projects}")
            
            if "shoprank" in projects:
                print("\n[CONCLUSION]")
                print("'shoprank' exists and is accessible by this Token and Account ID.")
                print("If GitHub Actions failed with 404, the Account ID or Token in GitHub Secrets does not match the ones you just tested.")
            else:
                print("\n[CONCLUSION]")
                print(f"'shoprank' is NOT in this account. The Account ID you provided is wrong.")
    except urllib.error.HTTPError as e:
        print(f" -> [FAILED] HTTP {e.code}. The token does not have access to Account {account_id}, or the Account ID is completely wrong.")

if __name__ == "__main__":
    main()
