import typing_extensions
from mallard_client.apis.paths.images_bucket_name import ImagesBucketName
from mallard_client.apis.paths.images_create_uav import ImagesCreateUav
from mallard_client.apis.paths.images_delete import ImagesDelete
from mallard_client.apis.paths.images_metadata import ImagesMetadata
from mallard_client.apis.paths.images_metadata_batch_update import (
    ImagesMetadataBatchUpdate,
)
from mallard_client.apis.paths.images_metadata_infer import ImagesMetadataInfer
from mallard_client.apis.paths.query import Query
from mallard_client.apis.paths.thumbnail_bucket_name import ThumbnailBucketName
from mallard_client.apis.paths.videos_bucket_name import VideosBucketName
from mallard_client.apis.paths.videos_create_uav import VideosCreateUav
from mallard_client.apis.paths.videos_delete import VideosDelete
from mallard_client.apis.paths.videos_metadata import VideosMetadata
from mallard_client.apis.paths.videos_metadata_batch_update import (
    VideosMetadataBatchUpdate,
)
from mallard_client.apis.paths.videos_metadata_infer import VideosMetadataInfer
from mallard_client.apis.paths.videos_preview_bucket_name import (
    VideosPreviewBucketName,
)
from mallard_client.apis.paths.videos_stream_bucket_name import (
    VideosStreamBucketName,
)
from mallard_client.paths import PathValues

PathToApi = typing_extensions.TypedDict(
    "PathToApi",
    {
        PathValues.IMAGES_CREATE_UAV: ImagesCreateUav,
        PathValues.IMAGES_DELETE: ImagesDelete,
        PathValues.IMAGES_BUCKET_NAME: ImagesBucketName,
        PathValues.IMAGES_METADATA: ImagesMetadata,
        PathValues.IMAGES_METADATA_BATCH_UPDATE: ImagesMetadataBatchUpdate,
        PathValues.IMAGES_METADATA_INFER: ImagesMetadataInfer,
        PathValues.VIDEOS_CREATE_UAV: VideosCreateUav,
        PathValues.VIDEOS_DELETE: VideosDelete,
        PathValues.VIDEOS_BUCKET_NAME: VideosBucketName,
        PathValues.VIDEOS_PREVIEW_BUCKET_NAME: VideosPreviewBucketName,
        PathValues.VIDEOS_STREAM_BUCKET_NAME: VideosStreamBucketName,
        PathValues.VIDEOS_METADATA: VideosMetadata,
        PathValues.VIDEOS_METADATA_BATCH_UPDATE: VideosMetadataBatchUpdate,
        PathValues.VIDEOS_METADATA_INFER: VideosMetadataInfer,
        PathValues.QUERY: Query,
        PathValues.THUMBNAIL_BUCKET_NAME: ThumbnailBucketName,
    },
)

path_to_api = PathToApi(
    {
        PathValues.IMAGES_CREATE_UAV: ImagesCreateUav,
        PathValues.IMAGES_DELETE: ImagesDelete,
        PathValues.IMAGES_BUCKET_NAME: ImagesBucketName,
        PathValues.IMAGES_METADATA: ImagesMetadata,
        PathValues.IMAGES_METADATA_BATCH_UPDATE: ImagesMetadataBatchUpdate,
        PathValues.IMAGES_METADATA_INFER: ImagesMetadataInfer,
        PathValues.VIDEOS_CREATE_UAV: VideosCreateUav,
        PathValues.VIDEOS_DELETE: VideosDelete,
        PathValues.VIDEOS_BUCKET_NAME: VideosBucketName,
        PathValues.VIDEOS_PREVIEW_BUCKET_NAME: VideosPreviewBucketName,
        PathValues.VIDEOS_STREAM_BUCKET_NAME: VideosStreamBucketName,
        PathValues.VIDEOS_METADATA: VideosMetadata,
        PathValues.VIDEOS_METADATA_BATCH_UPDATE: VideosMetadataBatchUpdate,
        PathValues.VIDEOS_METADATA_INFER: VideosMetadataInfer,
        PathValues.QUERY: Query,
        PathValues.THUMBNAIL_BUCKET_NAME: ThumbnailBucketName,
    }
)
