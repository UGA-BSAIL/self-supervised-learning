# mallard_client.ImagesApi

All URIs are relative to */api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**batch_update_metadata_images_metadata_batch_update_patch**](ImagesApi.md#batch_update_metadata_images_metadata_batch_update_patch) | **PATCH** /images/metadata/batch_update | Batch Update Metadata
[**create_uav_image_images_create_uav_post**](ImagesApi.md#create_uav_image_images_create_uav_post) | **POST** /images/create_uav | Create Uav Image
[**delete_images_images_delete_delete**](ImagesApi.md#delete_images_images_delete_delete) | **DELETE** /images/delete | Delete Images
[**find_image_metadata_images_metadata_post**](ImagesApi.md#find_image_metadata_images_metadata_post) | **POST** /images/metadata | Find Image Metadata
[**get_image_images_bucket_name_get**](ImagesApi.md#get_image_images_bucket_name_get) | **GET** /images/{bucket}/{name} | Get Image
[**infer_image_metadata_images_metadata_infer_post**](ImagesApi.md#infer_image_metadata_images_metadata_infer_post) | **POST** /images/metadata/infer | Infer Image Metadata


# **batch_update_metadata_images_metadata_batch_update_patch**
> object batch_update_metadata_images_metadata_batch_update_patch(body_batch_update_metadata_images_metadata_batch_update_patch, increment_sequence=increment_sequence, auth_token=auth_token)

Batch Update Metadata

Updates the metadata for a large number of images at once. Note that any parameters that are set to `None` in `metadata` will retain their original values.  Args:     metadata: The new metadata to set.     images: The set of existing images to update.     increment_sequence: If this is true, the sequence number will be         automatically incremented for each image added, starting at whatever         value is set in `metadata`. In this case, the order of the images         specified in `images` will determine session numbers.     metadata_store: The metadata store to use.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.body_batch_update_metadata_images_metadata_batch_update_patch import BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch
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
    api_instance = mallard_client.ImagesApi(api_client)
    body_batch_update_metadata_images_metadata_batch_update_patch = mallard_client.BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch() # BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch |
    increment_sequence = False # bool |  (optional) (default to False)
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Batch Update Metadata
        api_response = api_instance.batch_update_metadata_images_metadata_batch_update_patch(body_batch_update_metadata_images_metadata_batch_update_patch, increment_sequence=increment_sequence, auth_token=auth_token)
        print("The response of ImagesApi->batch_update_metadata_images_metadata_batch_update_patch:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ImagesApi->batch_update_metadata_images_metadata_batch_update_patch: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **body_batch_update_metadata_images_metadata_batch_update_patch** | [**BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch**](BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch.md)|  |
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

# **create_uav_image_images_create_uav_post**
> MallardGatewayRoutersImagesSchemasCreateResponse create_uav_image_images_create_uav_post(tz, image_data, auth_token=auth_token, size=size, name=name, platform_type=platform_type, notes=notes, session_name=session_name, sequence_number=sequence_number, capture_date=capture_date, location_description=location_description, camera=camera, altitude_meters=altitude_meters, gsd_cm_px=gsd_cm_px, format=format, latitude_deg=latitude_deg, longitude_deg=longitude_deg)

Create Uav Image

Uploads a new image captured from a UAV.  Args:     metadata: The image-specific metadata.     image_data: The actual image file to upload.     object_store: The object store to use.     metadata_store: The metadata store to use.     bucket: The bucket to use for new images.  Returns:     A `CreateResponse` object for this image.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.mallard_gateway_routers_images_schemas_create_response import MallardGatewayRoutersImagesSchemasCreateResponse
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
    api_instance = mallard_client.ImagesApi(api_client)
    tz = 3.4 # float |
    image_data = None # bytearray |
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
    latitude_deg = 3.4 # float |  (optional)
    longitude_deg = 3.4 # float |  (optional)

    try:
        # Create Uav Image
        api_response = api_instance.create_uav_image_images_create_uav_post(tz, image_data, auth_token=auth_token, size=size, name=name, platform_type=platform_type, notes=notes, session_name=session_name, sequence_number=sequence_number, capture_date=capture_date, location_description=location_description, camera=camera, altitude_meters=altitude_meters, gsd_cm_px=gsd_cm_px, format=format, latitude_deg=latitude_deg, longitude_deg=longitude_deg)
        print("The response of ImagesApi->create_uav_image_images_create_uav_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ImagesApi->create_uav_image_images_create_uav_post: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **tz** | **float**|  |
 **image_data** | **bytearray**|  |
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
 **latitude_deg** | **float**|  | [optional]
 **longitude_deg** | **float**|  | [optional]

### Return type

[**MallardGatewayRoutersImagesSchemasCreateResponse**](MallardGatewayRoutersImagesSchemasCreateResponse.md)

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

# **delete_images_images_delete_delete**
> object delete_images_images_delete_delete(object_ref, auth_token=auth_token)

Delete Images

Deletes existing images from the server.  Args:     images: The images to delete.     object_store: The object store to use.     metadata_store: The metadata store to use.

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
    api_instance = mallard_client.ImagesApi(api_client)
    object_ref = [mallard_client.ObjectRef()] # List[ObjectRef] |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Delete Images
        api_response = api_instance.delete_images_images_delete_delete(object_ref, auth_token=auth_token)
        print("The response of ImagesApi->delete_images_images_delete_delete:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ImagesApi->delete_images_images_delete_delete: %s\n" % e)
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

# **find_image_metadata_images_metadata_post**
> MallardGatewayRoutersImagesSchemasMetadataResponse find_image_metadata_images_metadata_post(object_ref, auth_token=auth_token)

Find Image Metadata

Retrieves the metadata for a set of images.  Args:     images: The set of images to get metadata for.     metadata_store: The metadata store to use.  Returns:     The corresponding metadata for each image, in JSON form.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.mallard_gateway_routers_images_schemas_metadata_response import MallardGatewayRoutersImagesSchemasMetadataResponse
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
    api_instance = mallard_client.ImagesApi(api_client)
    object_ref = [mallard_client.ObjectRef()] # List[ObjectRef] |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Find Image Metadata
        api_response = api_instance.find_image_metadata_images_metadata_post(object_ref, auth_token=auth_token)
        print("The response of ImagesApi->find_image_metadata_images_metadata_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ImagesApi->find_image_metadata_images_metadata_post: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **object_ref** | [**List[ObjectRef]**](ObjectRef.md)|  |
 **auth_token** | **str**|  | [optional]

### Return type

[**MallardGatewayRoutersImagesSchemasMetadataResponse**](MallardGatewayRoutersImagesSchemasMetadataResponse.md)

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

# **get_image_images_bucket_name_get**
> object get_image_images_bucket_name_get(bucket, name, auth_token=auth_token)

Get Image

Gets the contents of a specific image.  Args:     bucket: The bucket that the image is in.     name: The name of the image.     object_store: The object store to use.     metadata_store: The metadata store to use.  Returns:     The binary contents of the image.

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
    api_instance = mallard_client.ImagesApi(api_client)
    bucket = 'bucket_example' # str |
    name = 'name_example' # str |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Get Image
        api_response = api_instance.get_image_images_bucket_name_get(bucket, name, auth_token=auth_token)
        print("The response of ImagesApi->get_image_images_bucket_name_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ImagesApi->get_image_images_bucket_name_get: %s\n" % e)
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

# **infer_image_metadata_images_metadata_infer_post**
> UavImageMetadata infer_image_metadata_images_metadata_infer_post(tz, image_data, auth_token=auth_token, size=size, name=name, platform_type=platform_type, notes=notes, session_name=session_name, sequence_number=sequence_number, capture_date=capture_date, location_description=location_description, camera=camera, altitude_meters=altitude_meters, gsd_cm_px=gsd_cm_px, format=format, latitude_deg=latitude_deg, longitude_deg=longitude_deg)

Infer Image Metadata

Infers the metadata for an image.  Args:     metadata: Can be used to provide partial metadata to build on.  Returns:     The metadata that it was able to infer.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.uav_image_metadata import UavImageMetadata
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
    api_instance = mallard_client.ImagesApi(api_client)
    tz = 3.4 # float |
    image_data = None # bytearray |
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
    latitude_deg = 3.4 # float |  (optional)
    longitude_deg = 3.4 # float |  (optional)

    try:
        # Infer Image Metadata
        api_response = api_instance.infer_image_metadata_images_metadata_infer_post(tz, image_data, auth_token=auth_token, size=size, name=name, platform_type=platform_type, notes=notes, session_name=session_name, sequence_number=sequence_number, capture_date=capture_date, location_description=location_description, camera=camera, altitude_meters=altitude_meters, gsd_cm_px=gsd_cm_px, format=format, latitude_deg=latitude_deg, longitude_deg=longitude_deg)
        print("The response of ImagesApi->infer_image_metadata_images_metadata_infer_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling ImagesApi->infer_image_metadata_images_metadata_infer_post: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **tz** | **float**|  |
 **image_data** | **bytearray**|  |
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
 **latitude_deg** | **float**|  | [optional]
 **longitude_deg** | **float**|  | [optional]

### Return type

[**UavImageMetadata**](UavImageMetadata.md)

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
