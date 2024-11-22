import typing_extensions
from mallard_client.apis.tags import TagValues
from mallard_client.apis.tags.default_api import DefaultApi
from mallard_client.apis.tags.images_api import ImagesApi
from mallard_client.apis.tags.videos_api import VideosApi

TagToApi = typing_extensions.TypedDict(
    "TagToApi",
    {
        TagValues.DEFAULT: DefaultApi,
        TagValues.IMAGES: ImagesApi,
        TagValues.VIDEOS: VideosApi,
    },
)

tag_to_api = TagToApi(
    {
        TagValues.DEFAULT: DefaultApi,
        TagValues.IMAGES: ImagesApi,
        TagValues.VIDEOS: VideosApi,
    }
)
