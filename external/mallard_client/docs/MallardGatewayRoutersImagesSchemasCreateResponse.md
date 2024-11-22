# MallardGatewayRoutersImagesSchemasCreateResponse

Response to use for image creation requests.  Attributes:     image_id: The unique ID of the image that was created.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**image_id** | [**ObjectRef**](ObjectRef.md) |  |

## Example

```python
from mallard_client.models.mallard_gateway_routers_images_schemas_create_response import MallardGatewayRoutersImagesSchemasCreateResponse

# TODO update the JSON string below
json = "{}"
# create an instance of MallardGatewayRoutersImagesSchemasCreateResponse from a JSON string
mallard_gateway_routers_images_schemas_create_response_instance = MallardGatewayRoutersImagesSchemasCreateResponse.from_json(json)
# print the JSON string representation of the object
print MallardGatewayRoutersImagesSchemasCreateResponse.to_json()

# convert the object into a dict
mallard_gateway_routers_images_schemas_create_response_dict = mallard_gateway_routers_images_schemas_create_response_instance.to_dict()
# create an instance of MallardGatewayRoutersImagesSchemasCreateResponse from a dict
mallard_gateway_routers_images_schemas_create_response_form_dict = mallard_gateway_routers_images_schemas_create_response.from_dict(mallard_gateway_routers_images_schemas_create_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
