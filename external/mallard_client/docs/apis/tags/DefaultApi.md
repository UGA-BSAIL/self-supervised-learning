<a id="__pageTop"></a>
# mallard_client.apis.tags.default_api.DefaultApi

All URIs are relative to */api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_thumbnail_thumbnail_bucket_name_get**](#get_thumbnail_thumbnail_bucket_name_get) | **get** /thumbnail/{bucket}/{name} | Get Thumbnail
[**query_artifacts_query_post**](#query_artifacts_query_post) | **post** /query | Query Artifacts

# **get_thumbnail_thumbnail_bucket_name_get**
<a id="get_thumbnail_thumbnail_bucket_name_get"></a>
> bool, date, datetime, dict, float, int, list, str, none_type get_thumbnail_thumbnail_bucket_name_get(bucketname)

Get Thumbnail

Gets the thumbnail for a specific image.  Args:     bucket: The bucket that the image is in.     name: The name of the image.     object_store: The object store to use.  Returns:     The binary contents of the thumbnail.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import mallard_client
from mallard_client.apis.tags import default_api
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
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'bucket': "bucket_example",
        'name': "name_example",
    }
    query_params = {
    }
    try:
        # Get Thumbnail
        api_response = api_instance.get_thumbnail_thumbnail_bucket_name_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling DefaultApi->get_thumbnail_thumbnail_bucket_name_get: %s\n" % e)

    # example passing only optional values
    path_params = {
        'bucket': "bucket_example",
        'name': "name_example",
    }
    query_params = {
        'auth_token': "auth_token_example",
    }
    try:
        # Get Thumbnail
        api_response = api_instance.get_thumbnail_thumbnail_bucket_name_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling DefaultApi->get_thumbnail_thumbnail_bucket_name_get: %s\n" % e)
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
200 | [ApiResponseFor200](#get_thumbnail_thumbnail_bucket_name_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_thumbnail_thumbnail_bucket_name_get.ApiResponseFor422) | Validation Error

#### get_thumbnail_thumbnail_bucket_name_get.ApiResponseFor200
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

#### get_thumbnail_thumbnail_bucket_name_get.ApiResponseFor422
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

# **query_artifacts_query_post**
<a id="query_artifacts_query_post"></a>
> QueryResponse query_artifacts_query_post()

Query Artifacts

Performs a query for artifacts that meet certain criteria.  Args:     queries: Specifies the queries to perform.     orderings: Specifies a specific ordering for the final results. It         will first sort by the first ordering specified, then the         second, etc.     results_per_page: The maximum number of results to include in a         single response.     page_num: If there are multiple pages of results, this can be used to         specify a later page.     metadata_store: The metadata store to use.  Returns:     The query response.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import mallard_client
from mallard_client.apis.tags import default_api
from mallard_client.model.body_query_artifacts_query_post import BodyQueryArtifactsQueryPost
from mallard_client.model.query_response import QueryResponse
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
    api_instance = default_api.DefaultApi(api_client)

    # example passing only optional values
    query_params = {
        'results_per_page': 50,
        'page_num': 1,
        'auth_token': "auth_token_example",
    }
    body = BodyQueryArtifactsQueryPost(
        queries=[{}],
        orderings=[],
    )
    try:
        # Query Artifacts
        api_response = api_instance.query_artifacts_query_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except mallard_client.ApiException as e:
        print("Exception when calling DefaultApi->query_artifacts_query_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson, Unset] | optional, default is unset |
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
[**BodyQueryArtifactsQueryPost**](../../models/BodyQueryArtifactsQueryPost.md) |  |


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
results_per_page | ResultsPerPageSchema | | optional
page_num | PageNumSchema | | optional
auth_token | AuthTokenSchema | | optional


# ResultsPerPageSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | if omitted the server will use the default value of 50

# PageNumSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | if omitted the server will use the default value of 1

# AuthTokenSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  |

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#query_artifacts_query_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#query_artifacts_query_post.ApiResponseFor422) | Validation Error

#### query_artifacts_query_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**QueryResponse**](../../models/QueryResponse.md) |  |


#### query_artifacts_query_post.ApiResponseFor422
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
