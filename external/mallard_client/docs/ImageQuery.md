# ImageQuery

Represents a query for images that fit certain criteria. If multiple attributes are specified for this query, they will be ANDed. For instance, specifying both a name and sequence number range will look for images that both have a similar name, and have sequence numbers in that range.  Attributes:     platform_type: Search for data that was collected with this type of         robotic platform. Defaults to all types.     name: Partial-text search query for image names.     notes: Partial-text search query for image notes.     camera: Partial-text search query for camera models.      session: Partial-text search query for session names.     sequence_numbers: Look for images with these sequence numbers.     capture_dates: Look for images with these capture dates.      bounding_box: Geographic bounding box in which to look for data.     location_description: Partial-text search query for location         description.      altitude_meters: Look for images that were captured at these         altitudes, in meters AGL.     gsd_cm_px: Look for images that were captured with these ground sample         distances, in cm/px.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**platform_type** | [**PlatformType**](PlatformType.md) |  | [optional]
**name** | **str** |  | [optional]
**notes** | **str** |  | [optional]
**camera** | **str** |  | [optional]
**session** | **str** |  | [optional]
**sequence_numbers** | [**RangeInt**](RangeInt.md) |  | [optional]
**capture_dates** | [**RangeDate**](RangeDate.md) |  | [optional]
**bounding_box** | [**BoundingBox**](BoundingBox.md) |  | [optional]
**location_description** | **str** |  | [optional]
**altitude_meters** | [**RangeFloat**](RangeFloat.md) |  | [optional]
**gsd_cm_px** | [**RangeFloat**](RangeFloat.md) |  | [optional]

## Example

```python
from mallard_client.models.image_query import ImageQuery

# TODO update the JSON string below
json = "{}"
# create an instance of ImageQuery from a JSON string
image_query_instance = ImageQuery.from_json(json)
# print the JSON string representation of the object
print ImageQuery.to_json()

# convert the object into a dict
image_query_dict = image_query_instance.to_dict()
# create an instance of ImageQuery from a dict
image_query_form_dict = image_query.from_dict(image_query_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
