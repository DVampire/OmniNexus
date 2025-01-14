import os

from fastapi import HTTPException

from omninexus.core.logger import omninexus_logger as logger
from omninexus.server.types import AppMode, OmninexusConfigInterface
from omninexus.utils.import_utils import get_impl


class OmninexusConfig(OmninexusConfigInterface):
    config_cls = os.environ.get('OMNINEXUS_CONFIG_CLS', None)
    app_mode = AppMode.OSS
    posthog_client_key = 'phc_3ESMmY9SgqEAGBB6sMGK5ayYHkeUuknH2vP6FmWH9RA'
    github_client_id = os.environ.get('GITHUB_APP_CLIENT_ID', '')
    attach_conversation_middleware_path = (
        'omninexus.server.middleware.AttachConversationMiddleware'
    )
    settings_store_class: str = (
        'omninexus.storage.settings.file_settings_store.FileSettingsStore'
    )
    conversation_store_class: str = (
        'omninexus.storage.conversation.file_conversation_store.FileConversationStore'
    )

    def verify_config(self):
        if self.config_cls:
            raise ValueError('Unexpected config path provided')

    def verify_github_repo_list(self, installation_id: int | None):
        if self.app_mode == AppMode.OSS and installation_id:
            raise HTTPException(
                status_code=400,
                detail='Unexpected installation ID',
            )

    def get_config(self):
        config = {
            'APP_MODE': self.app_mode,
            'GITHUB_CLIENT_ID': self.github_client_id,
            'POSTHOG_CLIENT_KEY': self.posthog_client_key,
        }

        return config


def load_omninexus_config():
    config_cls = os.environ.get('OMNINEXUS_CONFIG_CLS', None)
    logger.info(f'Using config class {config_cls}')

    omninexus_config_cls = get_impl(OmninexusConfig, config_cls)
    omninexus_config = omninexus_config_cls()
    omninexus_config.verify_config()

    return omninexus_config
