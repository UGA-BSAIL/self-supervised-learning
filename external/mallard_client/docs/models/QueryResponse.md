# mallard_client.model.query_response.QueryResponse

Response to a query for images.  Attributes:     image_ids: The IDs of all images found by the query.      page_num: The page number that this query was for.     is_last_page: True if this represents the final page of query         results. Otherwise, there is at least one additional page. Note         that the last page might be empty.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Response to a query for images.  Attributes:     image_ids: The IDs of all images found by the query.      page_num: The page number that this query was for.     is_last_page: True if this represents the final page of query         results. Otherwise, there is at least one additional page. Note         that the last page might be empty. |

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**isLastPage** | bool,  | BoolClass,  |  |
**pageNum** | decimal.Decimal, int,  | decimal.Decimal,  |  |
**[imageIds](#imageIds)** | list, tuple,  | tuple,  |  |
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# imageIds

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  |

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**TypedObjectRef**](TypedObjectRef.md) | [**TypedObjectRef**](TypedObjectRef.md) | [**TypedObjectRef**](TypedObjectRef.md) |  |

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)
