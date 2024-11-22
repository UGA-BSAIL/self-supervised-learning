# MallardGatewayRoutersVideosSchemasMetadataResponse

Response to a request for image metadata.  Attributes:     metadata: The retrieved metadata for each image.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**metadata** | [**List[UavVideoMetadata]**](UavVideoMetadata.md) |  |

## Example

```python
from mallard_client.models.mallard_gateway_routers_videos_schemas_metadata_response import MallardGatewayRoutersVideosSchemasMetadataResponse

# TODO update the JSON string below
json = "{}"
# create an instance of MallardGatewayRoutersVideosSchemasMetadataResponse from a JSON string
mallard_gateway_routers_videos_schemas_metadata_response_instance = MallardGatewayRoutersVideosSchemasMetadataResponse.from_json(json)
# print the JSON string representation of the object
print MallardGatewayRoutersVideosSchemasMetadataResponse.to_json()

# convert the object into a dict
mallard_gateway_routers_videos_schemas_metadata_response_dict = mallard_gateway_routers_videos_schemas_metadata_response_instance.to_dict()
# create an instance of MallardGatewayRoutersVideosSchemasMetadataResponse from a dict
mallard_gateway_routers_videos_schemas_metadata_response_form_dict = mallard_gateway_routers_videos_schemas_metadata_response.from_dict(mallard_gateway_routers_videos_schemas_metadata_response_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
