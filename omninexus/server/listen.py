import socketio

from omninexus.server.app import app as base_app
from omninexus.server.listen_socket import sio
from omninexus.server.static import SPAStaticFiles

base_app.mount(
    '/', SPAStaticFiles(directory='./frontend/build', html=True), name='dist'
)

app = socketio.ASGIApp(sio, other_asgi_app=base_app)
