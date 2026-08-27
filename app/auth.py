from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


class AuthError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class User:
    id: int
    username: str
    role: str | None
    status: str
    must_change_password: bool
    is_admin: bool

    def public(self) -> dict[str, object]:
        return {"id": self.id, "username": self.username, "role": self.role, "status": self.status,
                "must_change_password": self.must_change_password, "is_admin": self.is_admin}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return "pbkdf2_sha256$310000$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt_value, digest_value = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt_value), int(rounds))
        return hmac.compare_digest(digest, base64.b64decode(digest_value))
    except (ValueError, TypeError):
        return False


def validate_password(password: str) -> None:
    if len(password) < 10 or not any(c.islower() for c in password) or not any(c.isupper() for c in password) or not any(c.isdigit() for c in password) or not any(not c.isalnum() for c in password):
        raise AuthError("密码至少 10 位，并同时包含大写字母、小写字母、数字和特殊字符。")


class AuthStore:
    def __init__(self, database_path: str, admin_initial_password: str):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.admin_initial_password = admin_initial_password
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL, role TEXT, status TEXT NOT NULL,
                    must_change_password INTEGER NOT NULL DEFAULT 1, is_admin INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                );
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, actor_id INTEGER, action TEXT NOT NULL,
                    target_user_id INTEGER, details TEXT, created_at TEXT NOT NULL
                );
            """)
            exists = db.execute("SELECT 1 FROM users WHERE username = 'admin'").fetchone()
            if not exists:
                timestamp = _now()
                db.execute("""INSERT INTO users(username,password_hash,role,status,must_change_password,is_admin,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?)""", ("admin", _hash_password(self.admin_initial_password), None, "active", 1, 1, timestamp, timestamp))

    @staticmethod
    def _user(row: sqlite3.Row | None) -> User | None:
        if not row:
            return None
        return User(row["id"], row["username"], row["role"], row["status"], bool(row["must_change_password"]), bool(row["is_admin"]))

    def register(self, username: str, password: str) -> User:
        username = username.strip().lower()
        if not 3 <= len(username) <= 32 or not all(char.isalnum() or char in "._-" for char in username):
            raise AuthError("用户名须为 3–32 位，仅可使用字母、数字、点、下划线或连字符。")
        validate_password(password)
        timestamp = _now()
        try:
            with self._connect() as db:
                cursor = db.execute("""INSERT INTO users(username,password_hash,role,status,must_change_password,is_admin,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?)""", (username, _hash_password(password), None, "pending", 1, 0, timestamp, timestamp))
                user_id = cursor.lastrowid
                db.execute("INSERT INTO audit_log(actor_id,action,target_user_id,created_at) VALUES(?,?,?,?)", (user_id, "registration_requested", user_id, timestamp))
        except sqlite3.IntegrityError as exc:
            raise AuthError("该用户名已被使用。") from exc
        return User(user_id, username, None, "pending", True, False)

    def authenticate(self, username: str, password: str) -> User:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),)).fetchone()
        if not row or not _verify_password(password, row["password_hash"]):
            raise AuthError("用户名或密码不正确。", 401)
        user = self._user(row)
        assert user
        if user.status == "disabled":
            raise AuthError("此账号已被停用，请联系管理员。", 403)
        return user

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        expiry = datetime.now(UTC) + timedelta(hours=8)
        with self._connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))
            db.execute("INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)", (hashlib.sha256(token.encode()).hexdigest(), user_id, expiry.isoformat()))
        return token

    def user_for_session(self, token: str | None) -> User | None:
        if not token:
            return None
        with self._connect() as db:
            row = db.execute("""SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id
                WHERE s.token_hash=? AND s.expires_at>?""", (hashlib.sha256(token.encode()).hexdigest(), _now())).fetchone()
        return self._user(row)

    def revoke_session(self, token: str | None) -> None:
        if token:
            with self._connect() as db:
                db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))

    def change_password(self, user: User, current_password: str, new_password: str) -> User:
        validate_password(new_password)
        if current_password == new_password:
            raise AuthError("新密码必须与当前密码不同。")
        with self._connect() as db:
            row = db.execute("SELECT password_hash FROM users WHERE id=?", (user.id,)).fetchone()
            if not row or not _verify_password(current_password, row["password_hash"]):
                raise AuthError("当前密码不正确。", 401)
            db.execute("UPDATE users SET password_hash=?,must_change_password=0,updated_at=? WHERE id=?", (_hash_password(new_password), _now(), user.id))
            db.execute("INSERT INTO audit_log(actor_id,action,target_user_id,created_at) VALUES(?,?,?,?)", (user.id, "password_changed", user.id, _now()))
        return self.get_user(user.id)

    def get_user(self, user_id: int) -> User:
        with self._connect() as db:
            user = self._user(db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())
        if not user:
            raise AuthError("账号不存在。", 404)
        return user

    def list_users(self) -> list[dict[str, object]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM users ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, created_at DESC").fetchall()
        return [self._user(row).public() | {"created_at": row["created_at"]} for row in rows if self._user(row)]

    def assign_access(self, admin: User, user_id: int, role: str, enabled: bool) -> User:
        target = self.get_user(user_id)
        if target.is_admin:
            raise AuthError("管理员账号不通过此流程分配业务角色。")
        status = "active" if enabled else "disabled"
        with self._connect() as db:
            db.execute("UPDATE users SET role=?,status=?,updated_at=? WHERE id=?", (role, status, _now(), user_id))
            db.execute("INSERT INTO audit_log(actor_id,action,target_user_id,details,created_at) VALUES(?,?,?,?,?)", (admin.id, "access_assigned", user_id, f"role={role};status={status}", _now()))
        return self.get_user(user_id)

    def audit_events(self) -> list[dict[str, object]]:
        with self._connect() as db:
            rows = db.execute("""SELECT a.action,a.details,a.created_at,actor.username actor,target.username target
                FROM audit_log a LEFT JOIN users actor ON actor.id=a.actor_id LEFT JOIN users target ON target.id=a.target_user_id
                ORDER BY a.id DESC LIMIT 30""").fetchall()
        return [dict(row) for row in rows]
