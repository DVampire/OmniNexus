from __future__ import annotations

import json
from dataclasses import dataclass

from omninexus.core.config.app_config import AppConfig
from omninexus.server.settings import Settings
from omninexus.storage import get_file_store
from omninexus.storage.files import FileStore
from omninexus.storage.settings.settings_store import SettingsStore
from omninexus.utils.async_utils import call_sync_from_async


@dataclass
class FileSettingsStore(SettingsStore):
    file_store: FileStore
    path: str = 'settings.json'

    async def load(self) -> Settings | None:
        try:
            json_str = await call_sync_from_async(self.file_store.read, self.path)
            kwargs = json.loads(json_str)
            settings = Settings(**kwargs)
            return settings
        except FileNotFoundError:
            return None

    async def store(self, settings: Settings):
        json_str = json.dumps(settings.__dict__)
        await call_sync_from_async(self.file_store.write, self.path, json_str)

    @classmethod
    async def get_instance(
        cls, config: AppConfig, user_id: int | None
    ) -> FileSettingsStore:
        file_store = get_file_store(config.file_store, config.file_store_path)
        return FileSettingsStore(file_store)
