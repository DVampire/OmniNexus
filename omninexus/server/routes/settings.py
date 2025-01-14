from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from omninexus.core.logger import omninexus_logger as logger
from omninexus.server.auth import get_user_id
from omninexus.server.settings import Settings
from omninexus.server.shared import config, omninexus_config
from omninexus.storage.conversation.conversation_store import ConversationStore
from omninexus.storage.settings.settings_store import SettingsStore
from omninexus.utils.import_utils import get_impl

app = APIRouter(prefix='/api')

SettingsStoreImpl = get_impl(SettingsStore, omninexus_config.settings_store_class)  # type: ignore
ConversationStoreImpl = get_impl(
    ConversationStore,  # type: ignore
    omninexus_config.conversation_store_class,
)


@app.get('/settings')
async def load_settings(request: Request) -> Settings | None:
    try:
        settings_store = await SettingsStoreImpl.get_instance(
            config, get_user_id(request)
        )
        settings = await settings_store.load()
        if not settings:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={'error': 'Settings not found'},
            )

        # For security reasons we don't ever send the api key to the client
        settings.llm_api_key = 'SET' if settings.llm_api_key else None
        return settings
    except Exception as e:
        logger.warning(f'Invalid token: {e}')
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Invalid token'},
        )


@app.post('/settings')
async def store_settings(
    request: Request,
    settings: Settings,
) -> JSONResponse:
    try:
        settings_store = await SettingsStoreImpl.get_instance(
            config, get_user_id(request)
        )
        existing_settings = await settings_store.load()

        if existing_settings:
            # LLM key isn't on the frontend, so we need to keep it if unset
            if settings.llm_api_key is None:
                settings.llm_api_key = existing_settings.llm_api_key

        # Update sandbox config with new settings
        if settings.remote_runtime_resource_factor is not None:
            config.sandbox.remote_runtime_resource_factor = (
                settings.remote_runtime_resource_factor
            )

        await settings_store.store(settings)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={'message': 'Settings stored'},
        )
    except Exception as e:
        logger.warning(f'Invalid token: {e}')
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={'error': 'Invalid token'},
        )
