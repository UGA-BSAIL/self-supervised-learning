# BoundingBox

Represents a location bounding box.  Attributes:     south_west: The south-west corner of the box.     north_east: The north-east corner of the box.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**south_west** | [**GeoPoint**](GeoPoint.md) |  |
**north_east** | [**GeoPoint**](GeoPoint.md) |  |

## Example

```python
from mallard_client.models.bounding_box import BoundingBox

# TODO update the JSON string below
json = "{}"
# create an instance of BoundingBox from a JSON string
bounding_box_instance = BoundingBox.from_json(json)
# print the JSON string representation of the object
print BoundingBox.to_json()

# convert the object into a dict
bounding_box_dict = bounding_box_instance.to_dict()
# create an instance of BoundingBox from a dict
bounding_box_form_dict = bounding_box.from_dict(bounding_box_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
