# UavImageMetadata

Represents metadata for an image.  Attributes:     format: The format that the image is in. This will be deduced         automatically, but an expected format can be provided by the user         for verification.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**size** | **int** |  | [optional]
**name** | **str** |  | [optional]
**platform_type** | [**PlatformType**](PlatformType.md) |  | [optional]
**notes** | **str** |  | [optional] [default to '']
**session_name** | **str** |  | [optional]
**sequence_number** | **int** |  | [optional]
**capture_date** | **date** |  | [optional]
**location** | [**GeoPoint**](GeoPoint.md) |  | [optional]
**location_description** | **str** |  | [optional]
**camera** | **str** |  | [optional]
**altitude_meters** | **float** |  | [optional]
**gsd_cm_px** | **float** |  | [optional]
**format** | [**ImageFormat**](ImageFormat.md) |  | [optional]

## Example

```python
from mallard_client.models.uav_image_metadata import UavImageMetadata

# TODO update the JSON string below
json = "{}"
# create an instance of UavImageMetadata from a JSON string
uav_image_metadata_instance = UavImageMetadata.from_json(json)
# print the JSON string representation of the object
print UavImageMetadata.to_json()

# convert the object into a dict
uav_image_metadata_dict = uav_image_metadata_instance.to_dict()
# create an instance of UavImageMetadata from a dict
uav_image_metadata_form_dict = uav_image_metadata.from_dict(uav_image_metadata_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
