<a id="__pageTop"></a>
# mallard_client.apis.tags.images_api.ImagesApi

All URIs are relative to */api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**batch_update_metadata_images_metadata_batch_update_patch**](#batch_update_metadata_images_metadata_batch_update_patch) | **patch** /images/metadata/batch_update | Batch Update Metadata
[**create_uav_image_images_create_uav_post**](#create_uav_image_images_create_uav_post) | **post** /images/create_uav | Create Uav Image
[**delete_images_images_delete_delete**](#delete_images_images_delete_delete) | **delete** /images/delete | Delete Images
[**find_image_metadata_images_metadata_post**](#find_image_metadata_images_metadata_post) | **post** /images/metadata | Find Image Metadata
[**get_image_images_bucket_name_get**](#get_image_images_bucket_name_get) | **get** /images/{bucket}/{name} | Get Image
[**infer_image_metadata_images_metadata_infer_post**](#infer_image_metadata_images_metadata_infer_post) | **post** /images/metadata/infer | Infer Image Metadata

# **batch_update_metadata_images_metadata_batch_update_patch**
<a id="batch_update_metadata_images_metadata_batch_update_patch"></a>
> bool, date, datetime, dict, float, int, list, str, none_type batch_update_metadata_images_metadata_batch_update_patch(body_batch_update_metadata_images_metadata_batch_update_patch)

Batch Update Metadata

Updates the metadata for a large number of images at once. Note that any parameters that are set to `None` in `metadata` will retain their original values.  Args:     metadata: The new metadata to set.     images: The set of existing images to update.     increment_sequence: If this is true, the sequence number will be         automatically incremented for each image added, starting at whatever         value is set in `metadata`. In this case, the order of the images         specified in `images` will determine session numbers.     metadata_store: The metadata store to use.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import mallard_client
from mallard_client.apis.tags import images_api
from mallard_client.model.body_batch_update_metadata_images_metadata_batch_update_patch import BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch
from mallard_client.model.http_validation_error import HTTPValidationError
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

# Configure OAuth2 access token for authorization: OAuth2AuthorizationCodeBearer
configuration = mallard_client.Configuration(
    host = "/api/v1",
    access_token = 'YOUR_ACCESS_TOKEN'
)
# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = images_api.ImagesApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
    }
    body = BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch(
        metadata=UavImageMetadata(
            size=1,
            name="name_example",
            platform_type=None,
            notes="",
            session_name="session_name_example",
            sequence_number=1,
            capture_date="1970-01-01",
            location=None,
            location_description="location_description_example",
            camera="camera_example",
            altitude_meters=3.14,
            gsd_cm_px=3.14,
            format=ImageFormat("gif"),
        ),
        images=[
            ObjectRef(
                bucket="bucket_example",
                name="name_example",
            )
        ],
    )
    try:
        # Batch Update Metadata
        api_response = api_instance.batch_update_metadata_images_metadata_batch_update_patch(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->batch_update_metadata_images_metadata_batch_update_patch: %s\n" % e)

    # example passing only optional values
    query_params = {
        'increment_sequence': False,
        'auth_token': "auth_token_example",
    }
    body = BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch(
        metadata=UavImageMetadata(
            size=1,
            name="name_example",
            platform_type=None,
            notes="",
            session_name="session_name_example",
            sequence_number=1,
            capture_date="1970-01-01",
            location=None,
            location_description="location_description_example",
            camera="camera_example",
            altitude_meters=3.14,
            gsd_cm_px=3.14,
            format=ImageFormat("gif"),
        ),
        images=[
            ObjectRef(
                bucket="bucket_example",
                name="name_example",
            )
        ],
    )
    try:
        # Batch Update Metadata
        api_response = api_instance.batch_update_metadata_images_metadata_batch_update_patch(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->batch_update_metadata_images_metadata_batch_update_patch: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch**](../../models/BodyBatchUpdateMetadataImagesMetadataBatchUpdatePatch.md) |  |


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
increment_sequence | IncrementSequenceSchema | | optional
auth_token | AuthTokenSchema | | optional


# IncrementSequenceSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
bool,  | BoolClass,  |  | if omitted the server will use the default value of False

# AuthTokenSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#batch_update_metadata_images_metadata_batch_update_patch.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#batch_update_metadata_images_metadata_batch_update_patch.ApiResponseFor422) | Validation Error

#### batch_update_metadata_images_metadata_batch_update_patch.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  |

#### batch_update_metadata_images_metadata_batch_update_patch.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  |


### Authorization

[OAuth2AuthorizationCodeBearer](../../../README.md#OAuth2AuthorizationCodeBearer)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_uav_image_images_create_uav_post**
<a id="create_uav_image_images_create_uav_post"></a>
> MallardGatewayRoutersImagesSchemasCreateResponse create_uav_image_images_create_uav_post(tz)

Create Uav Image

Uploads a new image captured from a UAV.  Args:     metadata: The image-specific metadata.     image_data: The actual image file to upload.     object_store: The object store to use.     metadata_store: The metadata store to use.     bucket: The bucket to use for new images.  Returns:     A `CreateResponse` object for this image.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import mallard_client
from mallard_client.apis.tags import images_api
from mallard_client.model.mallard_gateway_routers_images_schemas_create_response import MallardGatewayRoutersImagesSchemasCreateResponse
from mallard_client.model.body_create_uav_image_images_create_uav_post import BodyCreateUavImageImagesCreateUavPost
from mallard_client.model.http_validation_error import HTTPValidationError
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

# Configure OAuth2 access token for authorization: OAuth2AuthorizationCodeBearer
configuration = mallard_client.Configuration(
    host = "/api/v1",
    access_token = 'YOUR_ACCESS_TOKEN'
)
# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = images_api.ImagesApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'tz': -24.0,
    }
    try:
        # Create Uav Image
        api_response = api_instance.create_uav_image_images_create_uav_post(
            query_params=query_params,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->create_uav_image_images_create_uav_post: %s\n" % e)

    # example passing only optional values
    query_params = {
        'auth_token': "auth_token_example",
        'tz': -24.0,
    }
    body = dict(
        image_data=open('/path/to/file', 'rb'),
        size=1,
        name="name_example",
        platform_type="ground",
        notes="",
        session_name="session_name_example",
        sequence_number=1,
        capture_date="1970-01-01",
        location_description="location_description_example",
        camera="camera_example",
        altitude_meters=3.14,
        gsd_cm_px=3.14,
        format="format_example",
        latitude_deg=3.14,
        longitude_deg=3.14,
    )
    try:
        # Create Uav Image
        api_response = api_instance.create_uav_image_images_create_uav_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->create_uav_image_images_create_uav_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyMultipartFormData, Unset] | optional, default is unset |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'multipart/form-data' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyMultipartFormData
Type | Description  | Notes
------------- | ------------- | -------------
[**BodyCreateUavImageImagesCreateUavPost**](../../models/BodyCreateUavImageImagesCreateUavPost.md) |  |


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
auth_token | AuthTokenSchema | | optional
tz | TzSchema | |


# AuthTokenSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

# TzSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int, float,  | decimal.Decimal,  |  |

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
201 | [ApiResponseFor201](#create_uav_image_images_create_uav_post.ApiResponseFor201) | Successful Response
422 | [ApiResponseFor422](#create_uav_image_images_create_uav_post.ApiResponseFor422) | Validation Error

#### create_uav_image_images_create_uav_post.ApiResponseFor201
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor201ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor201ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**MallardGatewayRoutersImagesSchemasCreateResponse**](../../models/MallardGatewayRoutersImagesSchemasCreateResponse.md) |  |


#### create_uav_image_images_create_uav_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  |


### Authorization

[OAuth2AuthorizationCodeBearer](../../../README.md#OAuth2AuthorizationCodeBearer)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **delete_images_images_delete_delete**
<a id="delete_images_images_delete_delete"></a>
> bool, date, datetime, dict, float, int, list, str, none_type delete_images_images_delete_delete(object_ref)

Delete Images

Deletes existing images from the server.  Args:     images: The images to delete.     object_store: The object store to use.     metadata_store: The metadata store to use.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import mallard_client
from mallard_client.apis.tags import images_api
from mallard_client.model.object_ref import ObjectRef
from mallard_client.model.http_validation_error import HTTPValidationError
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

# Configure OAuth2 access token for authorization: OAuth2AuthorizationCodeBearer
configuration = mallard_client.Configuration(
    host = "/api/v1",
    access_token = 'YOUR_ACCESS_TOKEN'
)
# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = images_api.ImagesApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
    }
    body = [
        ObjectRef(
            bucket="bucket_example",
            name="name_example",
        )
    ]
    try:
        # Delete Images
        api_response = api_instance.delete_images_images_delete_delete(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->delete_images_images_delete_delete: %s\n" % e)

    # example passing only optional values
    query_params = {
        'auth_token': "auth_token_example",
    }
    body = [
        ObjectRef(
            bucket="bucket_example",
            name="name_example",
        )
    ]
    try:
        # Delete Images
        api_response = api_instance.delete_images_images_delete_delete(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->delete_images_images_delete_delete: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  |

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ObjectRef**]({{complexTypePrefix}}ObjectRef.md) | [**ObjectRef**]({{complexTypePrefix}}ObjectRef.md) | [**ObjectRef**]({{complexTypePrefix}}ObjectRef.md) |  |

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
auth_token | AuthTokenSchema | | optional


# AuthTokenSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#delete_images_images_delete_delete.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#delete_images_images_delete_delete.ApiResponseFor422) | Validation Error

#### delete_images_images_delete_delete.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  |

#### delete_images_images_delete_delete.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  |


### Authorization

[OAuth2AuthorizationCodeBearer](../../../README.md#OAuth2AuthorizationCodeBearer)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **find_image_metadata_images_metadata_post**
<a id="find_image_metadata_images_metadata_post"></a>
> MallardGatewayRoutersImagesSchemasMetadataResponse find_image_metadata_images_metadata_post(object_ref)

Find Image Metadata

Retrieves the metadata for a set of images.  Args:     images: The set of images to get metadata for.     metadata_store: The metadata store to use.  Returns:     The corresponding metadata for each image, in JSON form.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import mallard_client
from mallard_client.apis.tags import images_api
from mallard_client.model.mallard_gateway_routers_images_schemas_metadata_response import MallardGatewayRoutersImagesSchemasMetadataResponse
from mallard_client.model.object_ref import ObjectRef
from mallard_client.model.http_validation_error import HTTPValidationError
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

# Configure OAuth2 access token for authorization: OAuth2AuthorizationCodeBearer
configuration = mallard_client.Configuration(
    host = "/api/v1",
    access_token = 'YOUR_ACCESS_TOKEN'
)
# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = images_api.ImagesApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
    }
    body = [
        ObjectRef(
            bucket="bucket_example",
            name="name_example",
        )
    ]
    try:
        # Find Image Metadata
        api_response = api_instance.find_image_metadata_images_metadata_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->find_image_metadata_images_metadata_post: %s\n" % e)

    # example passing only optional values
    query_params = {
        'auth_token': "auth_token_example",
    }
    body = [
        ObjectRef(
            bucket="bucket_example",
            name="name_example",
        )
    ]
    try:
        # Find Image Metadata
        api_response = api_instance.find_image_metadata_images_metadata_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->find_image_metadata_images_metadata_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  |

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ObjectRef**]({{complexTypePrefix}}ObjectRef.md) | [**ObjectRef**]({{complexTypePrefix}}ObjectRef.md) | [**ObjectRef**]({{complexTypePrefix}}ObjectRef.md) |  |

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
auth_token | AuthTokenSchema | | optional


# AuthTokenSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#find_image_metadata_images_metadata_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#find_image_metadata_images_metadata_post.ApiResponseFor422) | Validation Error

#### find_image_metadata_images_metadata_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**MallardGatewayRoutersImagesSchemasMetadataResponse**](../../models/MallardGatewayRoutersImagesSchemasMetadataResponse.md) |  |


#### find_image_metadata_images_metadata_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  |


### Authorization

[OAuth2AuthorizationCodeBearer](../../../README.md#OAuth2AuthorizationCodeBearer)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_image_images_bucket_name_get**
<a id="get_image_images_bucket_name_get"></a>
> bool, date, datetime, dict, float, int, list, str, none_type get_image_images_bucket_name_get(bucketname)

Get Image

Gets the contents of a specific image.  Args:     bucket: The bucket that the image is in.     name: The name of the image.     object_store: The object store to use.     metadata_store: The metadata store to use.  Returns:     The binary contents of the image.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import mallard_client
from mallard_client.apis.tags import images_api
from mallard_client.model.http_validation_error import HTTPValidationError
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

# Configure OAuth2 access token for authorization: OAuth2AuthorizationCodeBearer
configuration = mallard_client.Configuration(
    host = "/api/v1",
    access_token = 'YOUR_ACCESS_TOKEN'
)
# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = images_api.ImagesApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'bucket': "bucket_example",
        'name': "name_example",
    }
    query_params = {
    }
    try:
        # Get Image
        api_response = api_instance.get_image_images_bucket_name_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->get_image_images_bucket_name_get: %s\n" % e)

    # example passing only optional values
    path_params = {
        'bucket': "bucket_example",
        'name': "name_example",
    }
    query_params = {
        'auth_token': "auth_token_example",
    }
    try:
        # Get Image
        api_response = api_instance.get_image_images_bucket_name_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->get_image_images_bucket_name_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
auth_token | AuthTokenSchema | | optional


# AuthTokenSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
bucket | BucketSchema | |
name | NameSchema | |

# BucketSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

# NameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_image_images_bucket_name_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_image_images_bucket_name_get.ApiResponseFor422) | Validation Error

#### get_image_images_bucket_name_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  |

#### get_image_images_bucket_name_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  |


### Authorization

[OAuth2AuthorizationCodeBearer](../../../README.md#OAuth2AuthorizationCodeBearer)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **infer_image_metadata_images_metadata_infer_post**
<a id="infer_image_metadata_images_metadata_infer_post"></a>
> UavImageMetadata infer_image_metadata_images_metadata_infer_post(tz)

Infer Image Metadata

Infers the metadata for an image.  Args:     metadata: Can be used to provide partial metadata to build on.  Returns:     The metadata that it was able to infer.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import mallard_client
from mallard_client.apis.tags import images_api
from mallard_client.model.uav_image_metadata import UavImageMetadata
from mallard_client.model.body_infer_image_metadata_images_metadata_infer_post import BodyInferImageMetadataImagesMetadataInferPost
from mallard_client.model.http_validation_error import HTTPValidationError
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

# Configure OAuth2 access token for authorization: OAuth2AuthorizationCodeBearer
configuration = mallard_client.Configuration(
    host = "/api/v1",
    access_token = 'YOUR_ACCESS_TOKEN'
)
# Enter a context with an instance of the API client
with mallard_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = images_api.ImagesApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'tz': -24.0,
    }
    try:
        # Infer Image Metadata
        api_response = api_instance.infer_image_metadata_images_metadata_infer_post(
            query_params=query_params,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->infer_image_metadata_images_metadata_infer_post: %s\n" % e)

    # example passing only optional values
    query_params = {
        'auth_token': "auth_token_example",
        'tz': -24.0,
    }
    body = dict(
        image_data=open('/path/to/file', 'rb'),
        size=1,
        name="name_example",
        platform_type="ground",
        notes="",
        session_name="session_name_example",
        sequence_number=1,
        capture_date="1970-01-01",
        location_description="location_description_example",
        camera="camera_example",
        altitude_meters=3.14,
        gsd_cm_px=3.14,
        format="format_example",
        latitude_deg=3.14,
        longitude_deg=3.14,
    )
    try:
        # Infer Image Metadata
        api_response = api_instance.infer_image_metadata_images_metadata_infer_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling ImagesApi->infer_image_metadata_images_metadata_infer_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyMultipartFormData, Unset] | optional, default is unset |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'multipart/form-data' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyMultipartFormData
Type | Description  | Notes
------------- | ------------- | -------------
[**BodyInferImageMetadataImagesMetadataInferPost**](../../models/BodyInferImageMetadataImagesMetadataInferPost.md) |  |


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
auth_token | AuthTokenSchema | | optional
tz | TzSchema | |


# AuthTokenSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

# TzSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int, float,  | decimal.Decimal,  |  |

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#infer_image_metadata_images_metadata_infer_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#infer_image_metadata_images_metadata_infer_post.ApiResponseFor422) | Validation Error

#### infer_image_metadata_images_metadata_infer_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UavImageMetadata**](../../models/UavImageMetadata.md) |  |


#### infer_image_metadata_images_metadata_infer_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  |


### Authorization

[OAuth2AuthorizationCodeBearer](../../../README.md#OAuth2AuthorizationCodeBearer)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)
