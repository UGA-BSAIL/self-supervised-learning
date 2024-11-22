# UavVideoMetadata

Represents metadata for a video.  Attributes:     format: The format that the video is in.      frame_rate: The video framerate, in FPS.     num_frames: The total number of frames in the video.

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
**format** | [**VideoFormat**](VideoFormat.md) |  | [optional]
**frame_rate** | **float** |  | [optional]
**num_frames** | **int** |  | [optional]

## Example

```python
from mallard_client.models.uav_video_metadata import UavVideoMetadata

# TODO update the JSON string below
json = "{}"
# create an instance of UavVideoMetadata from a JSON string
uav_video_metadata_instance = UavVideoMetadata.from_json(json)
# print the JSON string representation of the object
print UavVideoMetadata.to_json()

# convert the object into a dict
uav_video_metadata_dict = uav_video_metadata_instance.to_dict()
# create an instance of UavVideoMetadata from a dict
uav_video_metadata_form_dict = uav_video_metadata.from_dict(uav_video_metadata_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
