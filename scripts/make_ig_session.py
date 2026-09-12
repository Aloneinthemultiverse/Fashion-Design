"""Build an Instaloader session from a sessionid cookie you copy yourself.

`instaloader --load-cookies chrome` no longer works on Windows. Chrome 127 introduced
App-Bound Encryption, which ties the cookie key to Chrome itself so external tools cannot
decrypt the cookie database — the failure is `Unable to get key for cookie decryption`,
and browser_cookie3 has no way around it.

This takes the one cookie that matters and builds the session file directly. Nothing else
is needed: Instagram's web session is carried by `sessionid` alone.

The cookie never passes through the terminal or a command argument. Put it in a file:

    1. Open https://www.instagram.com in Chrome, logged in.
    2. F12 -> Application -> Storage -> Cookies -> https://www.instagram.com
    3. Find the row named `sessionid` and copy its Value.
    4. Save that value, on its own, into a file called `ig_sessionid.txt`
       in the project root.
    5. uv run python scripts/make_ig_session.py

`ig_sessionid.txt` and `ig.session` are both gitignored. Treat them like a password:
anyone holding that cookie is logged in as you until you log out, and logging out of
Instagram in the browser invalidates it everywhere, including here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cookie-file", type=Path, default=Path("ig_sessionid.txt"))
    parser.add_argument("--session-file", type=Path, default=Path("ig.session"))
    parser.add_argument(
        "--username",
        default="",
        help="Your Instagram username. Optional; only used to name the session.",
    )
    args = parser.parse_args()

    if not args.cookie_file.exists():
        raise SystemExit(
            f"{args.cookie_file} not found.\n\n"
            "Copy your sessionid cookie into it:\n"
            "  1. Open https://www.instagram.com in Chrome, logged in\n"
            "  2. F12 -> Application -> Cookies -> https://www.instagram.com\n"
            "  3. Copy the Value of the row named `sessionid`\n"
            f"  4. Paste it, alone, into {args.cookie_file}\n"
        )

    sessionid = args.cookie_file.read_text(encoding="utf-8").strip().strip('"')
    if not sessionid or len(sessionid) < 20:
        raise SystemExit(
            f"{args.cookie_file} does not look like a sessionid "
            f"(got {len(sessionid)} characters). Copy the full Value field."
        )

    try:
        import instaloader
    except ImportError:
        raise SystemExit("instaloader is not installed: uv pip install instaloader") from None

    loader = instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=False,
        download_comments=False,
        save_metadata=False,
        quiet=True,
    )
    loader.context._session.cookies.set("sessionid", sessionid, domain=".instagram.com")

    # Ask Instagram who we are. This both proves the cookie works and gets the username
    # instaloader keys the session by -- better than trusting a value typed by hand.
    try:
        username = loader.test_login()
    except Exception as exc:
        raise SystemExit(f"could not reach Instagram: {exc}") from exc

    if not username:
        raise SystemExit(
            "Instagram rejected that cookie. It is probably expired -- log out and back "
            "in at instagram.com, then copy the new sessionid."
        )

    loader.context.username = username
    loader.save_session_to_file(str(args.session_file))

    print(f"logged in as @{username}")
    print(f"session written to {args.session_file}")
    print("\nnow run:")
    print("  uv run python scripts/scrape_instagram.py --region indian --limit-celebrities 10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
