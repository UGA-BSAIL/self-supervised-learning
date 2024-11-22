# mallard_client.VideosApi

All URIs are relative to */api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**batch_update_metadata_videos_metadata_batch_update_patch**](VideosApi.md#batch_update_metadata_videos_metadata_batch_update_patch) | **PATCH** /videos/metadata/batch_update | Batch Update Metadata
[**create_uav_video_videos_create_uav_post**](VideosApi.md#create_uav_video_videos_create_uav_post) | **POST** /videos/create_uav | Create Uav Video
[**delete_videos_videos_delete_delete**](VideosApi.md#delete_videos_videos_delete_delete) | **DELETE** /videos/delete | Delete Videos
[**find_video_metadata_videos_metadata_post**](VideosApi.md#find_video_metadata_videos_metadata_post) | **POST** /videos/metadata | Find Video Metadata
[**get_preview_videos_preview_bucket_name_get**](VideosApi.md#get_preview_videos_preview_bucket_name_get) | **GET** /videos/preview/{bucket}/{name} | Get Preview
[**get_streamable_videos_stream_bucket_name_get**](VideosApi.md#get_streamable_videos_stream_bucket_name_get) | **GET** /videos/stream/{bucket}/{name} | Get Streamable
[**get_video_videos_bucket_name_get**](VideosApi.md#get_video_videos_bucket_name_get) | **GET** /videos/{bucket}/{name} | Get Video
[**infer_video_metadata_videos_metadata_infer_post**](VideosApi.md#infer_video_metadata_videos_metadata_infer_post) | **POST** /videos/metadata/infer | Infer Video Metadata


# **batch_update_metadata_videos_metadata_batch_update_patch**
> object batch_update_metadata_videos_metadata_batch_update_patch(body_batch_update_metadata_videos_metadata_batch_update_patch, increment_sequence=increment_sequence, auth_token=auth_token)

Batch Update Metadata

Updates the metadata for a large number of videos at once. Note that any parameters that are set to `None` in `metadata` will retain their original values.  Args:     metadata: The new metadata to set.     videos: The set of existing images to update.     increment_sequence: If this is true, the sequence number will be         automatically incremented for each image added, starting at whatever         value is set in `metadata`. In this case, the order of the images         specified in `images` will determine session numbers.     metadata_store: The metadata store to use.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.body_batch_update_metadata_videos_metadata_batch_update_patch import BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch
from mallard_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /api/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = mallard_client.Configuration(
    host = "/api/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

configuration.access_token = os.environ["ACCESS_TOKEN"]

# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mallard_client.VideosApi(api_client)
    body_batch_update_metadata_videos_metadata_batch_update_patch = mallard_client.BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch() # BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch |
    increment_sequence = False # bool |  (optional) (default to False)
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Batch Update Metadata
        api_response = api_instance.batch_update_metadata_videos_metadata_batch_update_patch(body_batch_update_metadata_videos_metadata_batch_update_patch, increment_sequence=increment_sequence, auth_token=auth_token)
        print("The response of VideosApi->batch_update_metadata_videos_metadata_batch_update_patch:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VideosApi->batch_update_metadata_videos_metadata_batch_update_patch: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body_batch_update_metadata_videos_metadata_batch_update_patch** | [**BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch**](BodyBatchUpdateMetadataVideosMetadataBatchUpdatePatch.md)|  |
 **increment_sequence** | **bool**|  | [optional] [default to False]
 **auth_token** | **str**|  | [optional]

### Return type

**object**

### Authorization

[OAuth2AuthorizationCodeBearer](../README.md#OAuth2AuthorizationCodeBearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **create_uav_video_videos_create_uav_post**
> MallardGatewayRoutersVideosSchemasCreateResponse create_uav_video_videos_create_uav_post(video_data, auth_token=auth_token, size=size, name=name, platform_type=platform_type, notes=notes, session_name=session_name, sequence_number=sequence_number, capture_date=capture_date, location_description=location_description, camera=camera, altitude_meters=altitude_meters, gsd_cm_px=gsd_cm_px, format=format, frame_rate=frame_rate, num_frames=num_frames, latitude_deg=latitude_deg, longitude_deg=longitude_deg)

Create Uav Video

Uploads a new video captured from a UAV.  Args:     metadata: The video-specific metadata.     video_data: The actual video file to upload.     object_store: The object store to upload the video to.     metadata_store: The metadata store to upload the metadata to.     bucket: The bucket to use for new videos.     background_tasks: Handle to use for submitting background tasks.  Returns:     A `CreateResponse` object for this video.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.mallard_gateway_routers_videos_schemas_create_response import MallardGatewayRoutersVideosSchemasCreateResponse
from mallard_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /api/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = mallard_client.Configuration(
    host = "/api/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

configuration.access_token = os.environ["ACCESS_TOKEN"]

# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mallard_client.VideosApi(api_client)
    video_data = None # bytearray |
    auth_token = 'auth_token_example' # str |  (optional)
    size = 56 # int |  (optional)
    name = 'name_example' # str |  (optional)
    platform_type = 'ground' # str |  (optional) (default to 'ground')
    notes = '' # str |  (optional) (default to '')
    session_name = 'session_name_example' # str |  (optional)
    sequence_number = 56 # int |  (optional)
    capture_date = '2013-10-20' # date |  (optional)
    location_description = 'location_description_example' # str |  (optional)
    camera = 'camera_example' # str |  (optional)
    altitude_meters = 3.4 # float |  (optional)
    gsd_cm_px = 3.4 # float |  (optional)
    format = 'format_example' # str |  (optional)
    frame_rate = 3.4 # float |  (optional)
    num_frames = 56 # int |  (optional)
    latitude_deg = 3.4 # float |  (optional)
    longitude_deg = 3.4 # float |  (optional)

    try:
        # Create Uav Video
        api_response = api_instance.create_uav_video_videos_create_uav_post(video_data, auth_token=auth_token, size=size, name=name, platform_type=platform_type, notes=notes, session_name=session_name, sequence_number=sequence_number, capture_date=capture_date, location_description=location_description, camera=camera, altitude_meters=altitude_meters, gsd_cm_px=gsd_cm_px, format=format, frame_rate=frame_rate, num_frames=num_frames, latitude_deg=latitude_deg, longitude_deg=longitude_deg)
        print("The response of VideosApi->create_uav_video_videos_create_uav_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VideosApi->create_uav_video_videos_create_uav_post: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **video_data** | **bytearray**|  |
 **auth_token** | **str**|  | [optional]
 **size** | **int**|  | [optional]
 **name** | **str**|  | [optional]
 **platform_type** | **str**|  | [optional] [default to &#39;ground&#39;]
 **notes** | **str**|  | [optional] [default to &#39;&#39;]
 **session_name** | **str**|  | [optional]
 **sequence_number** | **int**|  | [optional]
 **capture_date** | **date**|  | [optional]
 **location_description** | **str**|  | [optional]
 **camera** | **str**|  | [optional]
 **altitude_meters** | **float**|  | [optional]
 **gsd_cm_px** | **float**|  | [optional]
 **format** | **str**|  | [optional]
 **frame_rate** | **float**|  | [optional]
 **num_frames** | **int**|  | [optional]
 **latitude_deg** | **float**|  | [optional]
 **longitude_deg** | **float**|  | [optional]

### Return type

[**MallardGatewayRoutersVideosSchemasCreateResponse**](MallardGatewayRoutersVideosSchemasCreateResponse.md)

### Authorization

[OAuth2AuthorizationCodeBearer](../README.md#OAuth2AuthorizationCodeBearer)

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**201** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **delete_videos_videos_delete_delete**
> object delete_videos_videos_delete_delete(object_ref, auth_token=auth_token)

Delete Videos

Deletes existing videos from the server.  Args:     videos: The videos to delete.     object_store: The object store to delete the videos from.     metadata_store: The metadata store to delete the metadata from.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.object_ref import ObjectRef
from mallard_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /api/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = mallard_client.Configuration(
    host = "/api/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

configuration.access_token = os.environ["ACCESS_TOKEN"]

# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mallard_client.VideosApi(api_client)
    object_ref = [mallard_client.ObjectRef()] # List[ObjectRef] |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Delete Videos
        api_response = api_instance.delete_videos_videos_delete_delete(object_ref, auth_token=auth_token)
        print("The response of VideosApi->delete_videos_videos_delete_delete:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VideosApi->delete_videos_videos_delete_delete: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **object_ref** | [**List[ObjectRef]**](ObjectRef.md)|  |
 **auth_token** | **str**|  | [optional]

### Return type

**object**

### Authorization

[OAuth2AuthorizationCodeBearer](../README.md#OAuth2AuthorizationCodeBearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **find_video_metadata_videos_metadata_post**
> MallardGatewayRoutersVideosSchemasMetadataResponse find_video_metadata_videos_metadata_post(object_ref, auth_token=auth_token)

Find Video Metadata

Retrieves the metadata for a set of videos.  Args:     videos: The videos to retrieve the metadata for.     metadata_store: The metadata store to use.  Returns:     A `MetadataResponse` object containing the metadata for the videos.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.mallard_gateway_routers_videos_schemas_metadata_response import MallardGatewayRoutersVideosSchemasMetadataResponse
from mallard_client.models.object_ref import ObjectRef
from mallard_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /api/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = mallard_client.Configuration(
    host = "/api/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

configuration.access_token = os.environ["ACCESS_TOKEN"]

# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mallard_client.VideosApi(api_client)
    object_ref = [mallard_client.ObjectRef()] # List[ObjectRef] |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Find Video Metadata
        api_response = api_instance.find_video_metadata_videos_metadata_post(object_ref, auth_token=auth_token)
        print("The response of VideosApi->find_video_metadata_videos_metadata_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VideosApi->find_video_metadata_videos_metadata_post: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **object_ref** | [**List[ObjectRef]**](ObjectRef.md)|  |
 **auth_token** | **str**|  | [optional]

### Return type

[**MallardGatewayRoutersVideosSchemasMetadataResponse**](MallardGatewayRoutersVideosSchemasMetadataResponse.md)

### Authorization

[OAuth2AuthorizationCodeBearer](../README.md#OAuth2AuthorizationCodeBearer)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_preview_videos_preview_bucket_name_get**
> object get_preview_videos_preview_bucket_name_get(bucket, name, auth_token=auth_token)

Get Preview

Retrieves a preview from the server.  Args:     bucket: The bucket the video is in.     name: The name of the video.     object_store: The object store to use.  Returns:     A `StreamingResponse` object containing the thumbnail.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /api/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = mallard_client.Configuration(
    host = "/api/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

configuration.access_token = os.environ["ACCESS_TOKEN"]

# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mallard_client.VideosApi(api_client)
    bucket = 'bucket_example' # str |
    name = 'name_example' # str |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Get Preview
        api_response = api_instance.get_preview_videos_preview_bucket_name_get(bucket, name, auth_token=auth_token)
        print("The response of VideosApi->get_preview_videos_preview_bucket_name_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VideosApi->get_preview_videos_preview_bucket_name_get: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bucket** | **str**|  |
 **name** | **str**|  |
 **auth_token** | **str**|  | [optional]

### Return type

**object**

### Authorization

[OAuth2AuthorizationCodeBearer](../README.md#OAuth2AuthorizationCodeBearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_streamable_videos_stream_bucket_name_get**
> object get_streamable_videos_stream_bucket_name_get(bucket, name, auth_token=auth_token)

Get Streamable

Retrieves a streaming-optimized version of the video from the server.  Args:     bucket: The bucket the video is in.     name: The name of the video.     object_store: The object store to use.  Returns:     A `StreamingResponse` object containing the thumbnail.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /api/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = mallard_client.Configuration(
    host = "/api/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

configuration.access_token = os.environ["ACCESS_TOKEN"]

# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mallard_client.VideosApi(api_client)
    bucket = 'bucket_example' # str |
    name = 'name_example' # str |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Get Streamable
        api_response = api_instance.get_streamable_videos_stream_bucket_name_get(bucket, name, auth_token=auth_token)
        print("The response of VideosApi->get_streamable_videos_stream_bucket_name_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VideosApi->get_streamable_videos_stream_bucket_name_get: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bucket** | **str**|  |
 **name** | **str**|  |
 **auth_token** | **str**|  | [optional]

### Return type

**object**

### Authorization

[OAuth2AuthorizationCodeBearer](../README.md#OAuth2AuthorizationCodeBearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **get_video_videos_bucket_name_get**
> object get_video_videos_bucket_name_get(bucket, name, auth_token=auth_token)

Get Video

Retrieves a video from the server.  Args:     bucket: The bucket the video is in.     name: The name of the video.     object_store: The object store to retrieve the video from.     metadata_store: The metadata store to retrieve the metadata from.  Returns:     A `StreamingResponse` object containing the video.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /api/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = mallard_client.Configuration(
    host = "/api/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

configuration.access_token = os.environ["ACCESS_TOKEN"]

# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mallard_client.VideosApi(api_client)
    bucket = 'bucket_example' # str |
    name = 'name_example' # str |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Get Video
        api_response = api_instance.get_video_videos_bucket_name_get(bucket, name, auth_token=auth_token)
        print("The response of VideosApi->get_video_videos_bucket_name_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VideosApi->get_video_videos_bucket_name_get: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **bucket** | **str**|  |
 **name** | **str**|  |
 **auth_token** | **str**|  | [optional]

### Return type

**object**

### Authorization

[OAuth2AuthorizationCodeBearer](../README.md#OAuth2AuthorizationCodeBearer)

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **infer_video_metadata_videos_metadata_infer_post**
> UavVideoMetadata infer_video_metadata_videos_metadata_infer_post(video_data, auth_token=auth_token, size=size, name=name, platform_type=platform_type, notes=notes, session_name=session_name, sequence_number=sequence_number, capture_date=capture_date, location_description=location_description, camera=camera, altitude_meters=altitude_meters, gsd_cm_px=gsd_cm_px, format=format, frame_rate=frame_rate, num_frames=num_frames, latitude_deg=latitude_deg, longitude_deg=longitude_deg)

Infer Video Metadata

Infers the metadata for a video.  Args:     metadata: Can be used to provide partial metadata to build on.  Returns:     The metadata that it was able to infer.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.uav_video_metadata import UavVideoMetadata
from mallard_client.rest import ApiException
from pprint import pprint

# Defining the host is optional and defaults to /api/v1
# See configuration.py for a list of all supported configuration parameters.
configuration = mallard_client.Configuration(
    host = "/api/v1"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

configuration.access_token = os.environ["ACCESS_TOKEN"]

# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = mallard_client.VideosApi(api_client)
    video_data = None # bytearray |
    auth_token = 'auth_token_example' # str |  (optional)
    size = 56 # int |  (optional)
    name = 'name_example' # str |  (optional)
    platform_type = 'ground' # str |  (optional) (default to 'ground')
    notes = '' # str |  (optional) (default to '')
    session_name = 'session_name_example' # str |  (optional)
    sequence_number = 56 # int |  (optional)
    capture_date = '2013-10-20' # date |  (optional)
    location_description = 'location_description_example' # str |  (optional)
    camera = 'camera_example' # str |  (optional)
    altitude_meters = 3.4 # float |  (optional)
    gsd_cm_px = 3.4 # float |  (optional)
    format = 'format_example' # str |  (optional)
    frame_rate = 3.4 # float |  (optional)
    num_frames = 56 # int |  (optional)
    latitude_deg = 3.4 # float |  (optional)
    longitude_deg = 3.4 # float |  (optional)

    try:
        # Infer Video Metadata
        api_response = api_instance.infer_video_metadata_videos_metadata_infer_post(video_data, auth_token=auth_token, size=size, name=name, platform_type=platform_type, notes=notes, session_name=session_name, sequence_number=sequence_number, capture_date=capture_date, location_description=location_description, camera=camera, altitude_meters=altitude_meters, gsd_cm_px=gsd_cm_px, format=format, frame_rate=frame_rate, num_frames=num_frames, latitude_deg=latitude_deg, longitude_deg=longitude_deg)
        print("The response of VideosApi->infer_video_metadata_videos_metadata_infer_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling VideosApi->infer_video_metadata_videos_metadata_infer_post: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **video_data** | **bytearray**|  |
 **auth_token** | **str**|  | [optional]
 **size** | **int**|  | [optional]
 **name** | **str**|  | [optional]
 **platform_type** | **str**|  | [optional] [default to &#39;ground&#39;]
 **notes** | **str**|  | [optional] [default to &#39;&#39;]
 **session_name** | **str**|  | [optional]
 **sequence_number** | **int**|  | [optional]
 **capture_date** | **date**|  | [optional]
 **location_description** | **str**|  | [optional]
 **camera** | **str**|  | [optional]
 **altitude_meters** | **float**|  | [optional]
 **gsd_cm_px** | **float**|  | [optional]
 **format** | **str**|  | [optional]
 **frame_rate** | **float**|  | [optional]
 **num_frames** | **int**|  | [optional]
 **latitude_deg** | **float**|  | [optional]
 **longitude_deg** | **float**|  | [optional]

### Return type

[**UavVideoMetadata**](UavVideoMetadata.md)

### Authorization

[OAuth2AuthorizationCodeBearer](../README.md#OAuth2AuthorizationCodeBearer)

### HTTP request headers

 - **Content-Type**: multipart/form-data
 - **Accept**: application/json

### HTTP response details
| Status code | Description | Response headers |
|-------------|-------------|------------------|
**200** | Successful Response |  -  |
**422** | Validation Error |  -  |

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)
