import warnings
from contextlib import asynccontextmanager

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

from fastapi import (
    FastAPI,
)

import omninexus.agenthub  # noqa F401 (we import this to get the agents registered)
from omninexus.server.middleware import (
    AttachConversationMiddleware,
    GitHubTokenMiddleware,
    InMemoryRateLimiter,
    LocalhostCORSMiddleware,
    NoCacheMiddleware,
    RateLimitMiddleware,
)
from omninexus.server.routes.conversation import app as conversation_api_router
from omninexus.server.routes.feedback import app as feedback_api_router
from omninexus.server.routes.files import app as files_api_router
from omninexus.server.routes.github import app as github_api_router
from omninexus.server.routes.manage_conversations import (
    app as manage_conversation_api_router,
)
from omninexus.server.routes.public import app as public_api_router
from omninexus.server.routes.security import app as security_api_router
from omninexus.server.routes.settings import app as settings_router
from omninexus.server.shared import omninexus_config, session_manager
from omninexus.utils.import_utils import get_impl


@asynccontextmanager
async def _lifespan(app: FastAPI):
    async with session_manager:
        yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    LocalhostCORSMiddleware,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.add_middleware(GitHubTokenMiddleware)
app.add_middleware(NoCacheMiddleware)
app.add_middleware(
    RateLimitMiddleware, rate_limiter=InMemoryRateLimiter(requests=10, seconds=1)
)


@app.get('/health')
async def health():
    return 'OK'


app.include_router(public_api_router)
app.include_router(files_api_router)
app.include_router(security_api_router)
app.include_router(feedback_api_router)
app.include_router(conversation_api_router)
app.include_router(manage_conversation_api_router)
app.include_router(settings_router)
app.include_router(github_api_router)

AttachConversationMiddlewareImpl = get_impl(
    AttachConversationMiddleware, omninexus_config.attach_conversation_middleware_path
)
app.middleware('http')(AttachConversationMiddlewareImpl(app))
