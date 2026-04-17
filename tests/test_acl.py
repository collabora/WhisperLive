"""Tests for user management and access control (ACL)."""
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from whisper_live.acl import (
    Role,
    User,
    UserStore,
    JWTValidator,
    generate_api_key,
    _hash_key,
)


class TestRole(unittest.TestCase):
    def test_admin_can_do_everything(self):
        self.assertTrue(Role.ADMIN.can_transcribe())
        self.assertTrue(Role.ADMIN.can_admin())
        self.assertTrue(Role.ADMIN.can_read())

    def test_user_can_transcribe(self):
        self.assertTrue(Role.USER.can_transcribe())
        self.assertFalse(Role.USER.can_admin())
        self.assertTrue(Role.USER.can_read())

    def test_readonly_can_only_read(self):
        self.assertFalse(Role.READONLY.can_transcribe())
        self.assertFalse(Role.READONLY.can_admin())
        self.assertTrue(Role.READONLY.can_read())


class TestGenerateApiKey(unittest.TestCase):
    def test_key_format(self):
        key = generate_api_key()
        self.assertTrue(key.startswith("wl_"))
        self.assertGreater(len(key), 20)

    def test_keys_are_unique(self):
        keys = {generate_api_key() for _ in range(100)}
        self.assertEqual(len(keys), 100)


class TestUserSerialization(unittest.TestCase):
    def test_round_trip(self):
        user = User(
            user_id="abc123",
            name="Test User",
            role=Role.USER,
            api_key_hash="fakehash",
            rate_limit_rpm=120,
            quota_minutes=1000,
        )
        d = user.to_dict()
        self.assertEqual(d["role"], "user")
        restored = User.from_dict(d)
        self.assertEqual(restored.user_id, "abc123")
        self.assertEqual(restored.role, Role.USER)


class TestUserStore(unittest.TestCase):
    def setUp(self):
        self.tmpfile = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self.tmpfile.write("{}")
        self.tmpfile.close()
        self.store = UserStore(path=self.tmpfile.name)

    def tearDown(self):
        os.unlink(self.tmpfile.name)

    def test_create_and_authenticate(self):
        user, api_key = self.store.create_user("Alice", Role.USER)
        self.assertEqual(user.name, "Alice")
        self.assertTrue(api_key.startswith("wl_"))

        # Authenticate with correct key
        authed = self.store.authenticate(api_key)
        self.assertIsNotNone(authed)
        self.assertEqual(authed.user_id, user.user_id)

        # Fail with wrong key
        self.assertIsNone(self.store.authenticate("wrong-key"))

    def test_list_users(self):
        self.store.create_user("Alice", Role.USER)
        self.store.create_user("Bob", Role.ADMIN)
        users = self.store.list_users()
        self.assertEqual(len(users), 2)
        names = {u["name"] for u in users}
        self.assertEqual(names, {"Alice", "Bob"})

    def test_update_user(self):
        user, _ = self.store.create_user("Alice", Role.USER)
        updated = self.store.update_user(user.user_id, name="Alice Smith", rate_limit_rpm=200)
        self.assertEqual(updated.name, "Alice Smith")
        self.assertEqual(updated.rate_limit_rpm, 200)

    def test_delete_user(self):
        user, api_key = self.store.create_user("Alice", Role.USER)
        self.assertTrue(self.store.delete_user(user.user_id))
        self.assertIsNone(self.store.authenticate(api_key))
        self.assertFalse(self.store.delete_user("nonexistent"))

    def test_rotate_key(self):
        user, old_key = self.store.create_user("Alice", Role.USER)
        new_key = self.store.rotate_key(user.user_id)
        self.assertIsNotNone(new_key)
        self.assertNotEqual(old_key, new_key)

        # Old key no longer works
        self.assertIsNone(self.store.authenticate(old_key))
        # New key works
        self.assertIsNotNone(self.store.authenticate(new_key))

    def test_disabled_user_cannot_authenticate(self):
        user, api_key = self.store.create_user("Alice", Role.USER)
        self.store.update_user(user.user_id, enabled=False)
        self.assertIsNone(self.store.authenticate(api_key))

    def test_rate_limit(self):
        user, _ = self.store.create_user("Alice", Role.USER, rate_limit_rpm=3)
        # First 3 should pass
        for _ in range(3):
            self.assertTrue(self.store.check_rate_limit(user.user_id))
        # 4th should fail
        self.assertFalse(self.store.check_rate_limit(user.user_id))

    def test_quota(self):
        user, _ = self.store.create_user("Alice", Role.USER, quota_minutes=10)
        self.assertTrue(self.store.check_quota(user.user_id))
        self.store.track_usage(user.user_id, 10.0)
        self.assertFalse(self.store.check_quota(user.user_id))

    def test_reset_monthly_usage(self):
        user, _ = self.store.create_user("Alice", Role.USER, quota_minutes=10)
        self.store.track_usage(user.user_id, 9.5)
        self.store.reset_monthly_usage()
        self.assertTrue(self.store.check_quota(user.user_id))

    def test_persistence(self):
        """Users should survive store reload."""
        user, api_key = self.store.create_user("Alice", Role.USER)
        # Create a new store from the same file
        store2 = UserStore(path=self.tmpfile.name)
        authed = store2.authenticate(api_key)
        self.assertIsNotNone(authed)
        self.assertEqual(authed.name, "Alice")

    def test_unlimited_quota(self):
        user, _ = self.store.create_user("Alice", Role.USER, quota_minutes=0)
        self.store.track_usage(user.user_id, 999999.0)
        self.assertTrue(self.store.check_quota(user.user_id))  # 0 = unlimited


class TestJWTValidator(unittest.TestCase):
    def test_hs256_validation(self):
        """Test HS256 JWT validation with a shared secret."""
        try:
            import jwt as pyjwt
        except ImportError:
            self.skipTest("PyJWT not installed")

        secret = "test-secret-key"
        token = pyjwt.encode(
            {"sub": "user-1", "name": "Alice", "role": "user"},
            secret,
            algorithm="HS256",
        )

        validator = JWTValidator(secret=secret)
        claims = validator.validate(token)
        self.assertIsNotNone(claims)
        self.assertEqual(claims["sub"], "user-1")
        self.assertEqual(claims["name"], "Alice")

    def test_expired_token(self):
        try:
            import jwt as pyjwt
        except ImportError:
            self.skipTest("PyJWT not installed")

        secret = "test-secret"
        token = pyjwt.encode(
            {"sub": "user-1", "exp": time.time() - 100},
            secret,
            algorithm="HS256",
        )

        validator = JWTValidator(secret=secret)
        claims = validator.validate(token)
        self.assertIsNone(claims)

    def test_wrong_secret(self):
        try:
            import jwt as pyjwt
        except ImportError:
            self.skipTest("PyJWT not installed")

        token = pyjwt.encode({"sub": "user-1"}, "correct-secret", algorithm="HS256")
        validator = JWTValidator(secret="wrong-secret")
        claims = validator.validate(token)
        self.assertIsNone(claims)

    def test_get_user_info(self):
        validator = JWTValidator()
        claims = {
            "sub": "user-123",
            "email": "alice@example.com",
            "name": "Alice",
            "custom:role": "admin",
            "cognito:groups": ["admins", "users"],
        }
        info = validator.get_user_info(claims)
        self.assertEqual(info["user_id"], "user-123")
        self.assertEqual(info["email"], "alice@example.com")
        self.assertEqual(info["role"], "admin")


if __name__ == "__main__":
    unittest.main()
