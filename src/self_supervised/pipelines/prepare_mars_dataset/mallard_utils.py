"""
Utilities for enabling communication with MALLARD.
"""


import os
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin

import requests
from fief_client import Fief, FiefRequestError
from fief_client.integrations.cli import (
    FiefAuth,
    FiefAuthNotAuthenticatedError,
)
from loguru import logger
from mallard_client import Configuration, ObjectRef
from requests_toolbelt.multipart import (
    MultipartEncoder,
    MultipartEncoderMonitor,
)

_FIEF_URL = "https://bsailn2.ad.ufl.edu:8085"
"""
Fief URL to use for authentication.
"""
_MALLARD_URL = "https://bsailn2.ad.ufl.edu:8083/api/v1/"
"""
The MALLARD API base URL.
"""
_CONFIG_DIR = Path(__file__).parents[4] / "conf" / "local"
"""
Location of the directory to use for storing config files.
"""


def get_auth() -> FiefAuth:
    """
    Returns:
        A FiefAuth instance, properly configured.

    """
    client_id = (_CONFIG_DIR / "credentials_client_id.txt").read_text()
    client_id = client_id.strip("\n")
    fief = Fief(_FIEF_URL, client_id, verify=False)

    credentials_path = _CONFIG_DIR / "credentials.json"
    return FiefAuth(fief, credentials_path.as_posix())


def ensure_user_logged_in() -> None:
    """
    Ensures that the user is logged into MALLARD.

    """
    auth = get_auth()

    while True:
        try:
            user_info = auth.current_user()
            logger.info(f"User {user_info['email']} is logged in.")

            # This might fail if the access token has expired.
            auth.access_token_info()
            break
        except (FiefAuthNotAuthenticatedError, FiefRequestError):
            # Force the user to log in.
            auth.authorize()


def get_config() -> Configuration:
    """
    Gets the configuration to use for connecting to the MALLARD API.

    Returns:
        The configuration.

    """
    auth = get_auth()
    access_token_info = auth.access_token_info()
    access_token = access_token_info["access_token"]

    config = Configuration(host=_MALLARD_URL, access_token=access_token)
    # The server currently has a self-signed certificate.
    config.verify_ssl = False

    return config


def streaming_video_upload(
    video_file: Path, on_progress: Optional[Callable[[float], None]] = None
) -> ObjectRef:
    """
    Uploads a video through a streaming request, which openapi-generator doesn't
    natively support.

    Args:
        video_file: The file to upload.
        on_progress: If provided, it will be called regularly with the
            current fractional completion of the upload.

    Returns:
        The ID of the object it uploaded.

    """
    video_size = os.path.getsize(video_file.absolute().as_posix())

    def _on_progress(monitor: MultipartEncoderMonitor) -> None:
        if on_progress is not None:
            on_progress(monitor.bytes_read / video_size)

    # Generate the form.
    form_encoder = MultipartEncoder(
        dict(
            name=video_file.name,
            size=str(video_size),
            video_data=(video_file.name, open(video_file, "rb"), "video/H264"),
        )
    )
    monitored_encoder = MultipartEncoderMonitor(form_encoder, _on_progress)

    endpoint = urljoin(_MALLARD_URL, "videos/create_uav")
    token = get_auth().access_token_info()["access_token"]
    response = requests.post(
        endpoint,
        data=monitored_encoder,
        headers={
            "Content-Type": form_encoder.content_type,
            "Authorization": f"Bearer {token}",
        },
        verify=False,
        # Give it an hour to upload large files.
        timeout=(6.1, 60 * 60),
    )
    response.raise_for_status()

    # Parse the response.
    response_json = response.json()
    video_id = response_json["videoId"]
    return ObjectRef(bucket=video_id["bucket"], name=video_id["name"])


def streaming_video_download(video_url: str, destination: Path) -> None:
    """
    Downloads a video through a streaming request, which openapi-generator
    doesn't natively support.

    Args:
        video_url: The URL of the video.
        destination: The file to download to.

    """
    token = get_auth().access_token_info()["access_token"]
    response = requests.get(
        video_url,
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
        # Give it an hour to download large files.
        timeout=(6.1, 60 * 60),
        stream=True,
    )
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
