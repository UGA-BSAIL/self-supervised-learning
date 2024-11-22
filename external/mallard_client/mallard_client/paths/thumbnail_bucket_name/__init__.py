# do not import all endpoints into this module because that uses a lot of memory and stack frames
# if you need the ability to import all endpoints from this module, import them with
# from mallard_client.paths.thumbnail_bucket_name import Api

from mallard_client.paths import PathValues

path = PathValues.THUMBNAIL_BUCKET_NAME
