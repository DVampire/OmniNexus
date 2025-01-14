import os

import socketio
from dotenv import load_dotenv

from omninexus.core.config import load_app_config
from omninexus.server.config.omninexus_config import load_omninexus_config
from omninexus.server.session import SessionManager
from omninexus.storage import get_file_store

load_dotenv()

config = load_app_config()
omninexus_config = load_omninexus_config()
file_store = get_file_store(config.file_store, config.file_store_path)

client_manager = None
redis_host = os.environ.get('REDIS_HOST')
if redis_host:
    client_manager = socketio.AsyncRedisManager(
        f'redis://{redis_host}',
        redis_options={'password': os.environ.get('REDIS_PASSWORD')},
    )


sio = socketio.AsyncServer(
    async_mode='asgi', cors_allowed_origins='*', client_manager=client_manager
)

session_manager = SessionManager(sio, config, file_store)
