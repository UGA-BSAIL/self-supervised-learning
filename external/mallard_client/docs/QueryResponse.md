# QueryResponse

Response to a query for images.  Attributes:     image_ids: The IDs of all images found by the query.      page_num: The page number that this query was for.     is_last_page: True if this represents the final page of query         results. Otherwise, there is at least one additional page. Note         that the last page might be empty.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**image_ids** | [**List[TypedObjectRef]**](TypedObjectRef.md) |  |
**page_num** | **int** |  |
**is_last_page** | **bool** |  |

## Example

```python
from mallard_client.models.query_response import QueryResponse

# TODO update the JSON string below
json = "{}"
# create an instance of QueryResponse from a JSON string
query_response_instance = QueryResponse.from_json(json)
# print the JSON string representation of the object
print QueryResponse.to_json()

# convert the object into a dict
query_response_dict = query_response_instance.to_dict()
# create an instance of QueryResponse from a dict
query_response_form_dict = query_response.from_dict(query_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
