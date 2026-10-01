"""Admin command line (run inside the backend container).

    python -m sentinel.manage list-users
    python -m sentinel.manage reset-password <username> [--password P]   # prompts if omitted
    python -m sentinel.manage create-user <username> <admin|operator|viewer> [--password P]
"""
from __future__ import annotations

import argparse
import getpass
import sys

from sqlalchemy import select

from sentinel.auth.security import Role, hash_password
from sentinel.config import get_settings
from sentinel.db import Database, Tenant, User


def _password(given: str | None) -> str:
    if given:
        pw = given
    else:
        pw = getpass.getpass("New password: ")
        if pw != getpass.getpass("Repeat password: "):
            sys.exit("Passwords do not match.")
    if len(pw) < 10:
        sys.exit("Password must be at least 10 characters.")
    return pw


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m sentinel.manage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-users")
    r = sub.add_parser("reset-password")
    r.add_argument("username")
    r.add_argument("--password")
    c = sub.add_parser("create-user")
    c.add_argument("username")
    c.add_argument("role", choices=[x.value for x in Role])
    c.add_argument("--password")
    args = ap.parse_args(argv)

    s = get_settings()
    db = Database(s.database_url)
    db.init()
    with db.session() as ses:
        if args.cmd == "list-users":
            for u in ses.scalars(select(User).order_by(User.username)):
                print(f"{u.username:20s} {u.role:9s} {'active' if u.is_active else 'disabled'}")
            return
        user = ses.scalars(select(User).where(User.username == args.username)).first()
        if args.cmd == "reset-password":
            if user is None:
                sys.exit(f"No user '{args.username}'. Use list-users or create-user.")
            user.password_hash = hash_password(_password(args.password))
            user.is_active = True
            ses.commit()
            print(f"Password for '{user.username}' reset.")
        elif args.cmd == "create-user":
            if user is not None:
                sys.exit(f"User '{args.username}' already exists.")
            if ses.get(Tenant, s.tenant) is None:
                ses.add(Tenant(id=s.tenant, name=s.tenant.title()))
            ses.add(User(tenant_id=s.tenant, username=args.username, role=args.role,
                         password_hash=hash_password(_password(args.password))))
            ses.commit()
            print(f"Created {args.role} '{args.username}'.")


if __name__ == "__main__":
    main()
