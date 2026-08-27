from app.auth import AuthError, AuthStore, User


def test_registration_requires_approval_and_password_rotation(tmp_path):
    store = AuthStore(str(tmp_path / "accounts.sqlite"), "Admin@123456")
    user = store.register("alice", "InitialPass!2026")
    assert user.status == "pending" and user.must_change_password
    assert store.authenticate("alice", "InitialPass!2026").must_change_password
    changed = store.change_password(user, "InitialPass!2026", "ChangedPass!2026")
    assert not changed.must_change_password
    approved = store.assign_access(User(1, "admin", None, "active", False, True), changed.id, "engineering", True)
    assert approved.status == "active" and approved.role == "engineering"


def test_password_policy_rejects_weak_password(tmp_path):
    store = AuthStore(str(tmp_path / "accounts.sqlite"), "Admin@123456")
    try:
        store.register("alice", "weak")
    except AuthError as error:
        assert "至少 10 位" in str(error)
    else:
        raise AssertionError("weak password was accepted")
