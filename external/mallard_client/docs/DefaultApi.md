# mallard_client.DefaultApi

All URIs are relative to */api/v1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**get_thumbnail_thumbnail_bucket_name_get**](DefaultApi.md#get_thumbnail_thumbnail_bucket_name_get) | **GET** /thumbnail/{bucket}/{name} | Get Thumbnail
[**query_artifacts_query_post**](DefaultApi.md#query_artifacts_query_post) | **POST** /query | Query Artifacts


# **get_thumbnail_thumbnail_bucket_name_get**
> object get_thumbnail_thumbnail_bucket_name_get(bucket, name, auth_token=auth_token)

Get Thumbnail

Gets the thumbnail for a specific image.  Args:     bucket: The bucket that the image is in.     name: The name of the image.     object_store: The object store to use.  Returns:     The binary contents of the thumbnail.

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
    api_instance = mallard_client.DefaultApi(api_client)
    bucket = 'bucket_example' # str |
    name = 'name_example' # str |
    auth_token = 'auth_token_example' # str |  (optional)

    try:
        # Get Thumbnail
        api_response = api_instance.get_thumbnail_thumbnail_bucket_name_get(bucket, name, auth_token=auth_token)
        print("The response of DefaultApi->get_thumbnail_thumbnail_bucket_name_get:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->get_thumbnail_thumbnail_bucket_name_get: %s\n" % e)
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

# **query_artifacts_query_post**
> QueryResponse query_artifacts_query_post(results_per_page=results_per_page, page_num=page_num, auth_token=auth_token, body_query_artifacts_query_post=body_query_artifacts_query_post)

Query Artifacts

Performs a query for artifacts that meet certain criteria.  Args:     queries: Specifies the queries to perform.     orderings: Specifies a specific ordering for the final results. It         will first sort by the first ordering specified, then the         second, etc.     results_per_page: The maximum number of results to include in a         single response.     page_num: If there are multiple pages of results, this can be used to         specify a later page.     metadata_store: The metadata store to use.  Returns:     The query response.

### Example

* OAuth Authentication (OAuth2AuthorizationCodeBearer):
```python
import time
import os
import mallard_client
from mallard_client.models.body_query_artifacts_query_post import BodyQueryArtifactsQueryPost
from mallard_client.models.query_response import QueryResponse
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
    api_instance = mallard_client.DefaultApi(api_client)
    results_per_page = 50 # int |  (optional) (default to 50)
    page_num = 1 # int |  (optional) (default to 1)
    auth_token = 'auth_token_example' # str |  (optional)
    body_query_artifacts_query_post = mallard_client.BodyQueryArtifactsQueryPost() # BodyQueryArtifactsQueryPost |  (optional)

    try:
        # Query Artifacts
        api_response = api_instance.query_artifacts_query_post(results_per_page=results_per_page, page_num=page_num, auth_token=auth_token, body_query_artifacts_query_post=body_query_artifacts_query_post)
        print("The response of DefaultApi->query_artifacts_query_post:\n")
        pprint(api_response)
    except Exception as e:
        print("Exception when calling DefaultApi->query_artifacts_query_post: %s\n" % e)
```



### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **results_per_page** | **int**|  | [optional] [default to 50]
 **page_num** | **int**|  | [optional] [default to 1]
 **auth_token** | **str**|  | [optional]
 **body_query_artifacts_query_post** | [**BodyQueryArtifactsQueryPost**](BodyQueryArtifactsQueryPost.md)|  | [optional]

### Return type

[**QueryResponse**](QueryResponse.md)

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
