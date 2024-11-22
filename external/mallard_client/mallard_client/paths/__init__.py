# do not import all endpoints into this module because that uses a lot of memory and stack frames
# if you need the ability to import all endpoints from this module, import them with
# from mallard_client.apis.path_to_api import path_to_api

import enum


class PathValues(str, enum.Enum):
    IMAGES_CREATE_UAV = "/images/create_uav"
    IMAGES_DELETE = "/images/delete"
    IMAGES_BUCKET_NAME = "/images/{bucket}/{name}"
    IMAGES_METADATA = "/images/metadata"
    IMAGES_METADATA_BATCH_UPDATE = "/images/metadata/batch_update"
    IMAGES_METADATA_INFER = "/images/metadata/infer"
    VIDEOS_CREATE_UAV = "/videos/create_uav"
    VIDEOS_DELETE = "/videos/delete"
    VIDEOS_BUCKET_NAME = "/videos/{bucket}/{name}"
    VIDEOS_PREVIEW_BUCKET_NAME = "/videos/preview/{bucket}/{name}"
    VIDEOS_STREAM_BUCKET_NAME = "/videos/stream/{bucket}/{name}"
    VIDEOS_METADATA = "/videos/metadata"
    VIDEOS_METADATA_BATCH_UPDATE = "/videos/metadata/batch_update"
    VIDEOS_METADATA_INFER = "/videos/metadata/infer"
    QUERY = "/query"
    THUMBNAIL_BUCKET_NAME = "/thumbnail/{bucket}/{name}"
