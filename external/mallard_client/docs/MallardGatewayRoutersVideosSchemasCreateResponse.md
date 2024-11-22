# MallardGatewayRoutersVideosSchemasCreateResponse

Response to use for video creation requests.  Attributes:     video_id: The ID of the video that was created.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**video_id** | [**ObjectRef**](ObjectRef.md) |  |

## Example

```python
from mallard_client.models.mallard_gateway_routers_videos_schemas_create_response import MallardGatewayRoutersVideosSchemasCreateResponse

# TODO update the JSON string below
json = "{}"
# create an instance of MallardGatewayRoutersVideosSchemasCreateResponse from a JSON string
mallard_gateway_routers_videos_schemas_create_response_instance = MallardGatewayRoutersVideosSchemasCreateResponse.from_json(json)
# print the JSON string representation of the object
print MallardGatewayRoutersVideosSchemasCreateResponse.to_json()

# convert the object into a dict
mallard_gateway_routers_videos_schemas_create_response_dict = mallard_gateway_routers_videos_schemas_create_response_instance.to_dict()
# create an instance of MallardGatewayRoutersVideosSchemasCreateResponse from a dict
mallard_gateway_routers_videos_schemas_create_response_form_dict = mallard_gateway_routers_videos_schemas_create_response.from_dict(mallard_gateway_routers_videos_schemas_create_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
