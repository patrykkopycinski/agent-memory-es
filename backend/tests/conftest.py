import os

# Single session-wide key store: modules must not fight over AMES_API_KEYS_FILE
# (pytest imports all test modules before running any test).
os.environ["AMES_API_KEYS_FILE"] = "/tmp/ames_session_keys.json"
if os.path.exists(os.environ["AMES_API_KEYS_FILE"]):
    os.remove(os.environ["AMES_API_KEYS_FILE"])
