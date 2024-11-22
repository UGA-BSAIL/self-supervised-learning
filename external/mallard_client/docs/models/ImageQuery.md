# mallard_client.model.image_query.ImageQuery

Represents a query for images that fit certain criteria. If multiple attributes are specified for this query, they will be ANDed. For instance, specifying both a name and sequence number range will look for images that both have a similar name, and have sequence numbers in that range.  Attributes:     platform_type: Search for data that was collected with this type of         robotic platform. Defaults to all types.     name: Partial-text search query for image names.     notes: Partial-text search query for image notes.     camera: Partial-text search query for camera models.      session: Partial-text search query for session names.     sequence_numbers: Look for images with these sequence numbers.     capture_dates: Look for images with these capture dates.      bounding_box: Geographic bounding box in which to look for data.     location_description: Partial-text search query for location         description.      altitude_meters: Look for images that were captured at these         altitudes, in meters AGL.     gsd_cm_px: Look for images that were captured with these ground sample         distances, in cm/px.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Represents a query for images that fit certain criteria. If multiple attributes are specified for this query, they will be ANDed. For instance, specifying both a name and sequence number range will look for images that both have a similar name, and have sequence numbers in that range.  Attributes:     platform_type: Search for data that was collected with this type of         robotic platform. Defaults to all types.     name: Partial-text search query for image names.     notes: Partial-text search query for image notes.     camera: Partial-text search query for camera models.      session: Partial-text search query for session names.     sequence_numbers: Look for images with these sequence numbers.     capture_dates: Look for images with these capture dates.      bounding_box: Geographic bounding box in which to look for data.     location_description: Partial-text search query for location         description.      altitude_meters: Look for images that were captured at these         altitudes, in meters AGL.     gsd_cm_px: Look for images that were captured with these ground sample         distances, in cm/px. |

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**platformType** | [**PlatformType**](PlatformType.md) | [**PlatformType**](PlatformType.md) |  | [optional]
**name** | str,  | str,  |  | [optional]
**notes** | str,  | str,  |  | [optional]
**camera** | str,  | str,  |  | [optional]
**session** | str,  | str,  |  | [optional]
**sequenceNumbers** | [**RangeInt**](RangeInt.md) | [**RangeInt**](RangeInt.md) |  | [optional]
**captureDates** | [**RangeDate**](RangeDate.md) | [**RangeDate**](RangeDate.md) |  | [optional]
**boundingBox** | [**BoundingBox**](BoundingBox.md) | [**BoundingBox**](BoundingBox.md) |  | [optional]
**locationDescription** | str,  | str,  |  | [optional]
**altitudeMeters** | [**RangeFloat**](RangeFloat.md) | [**RangeFloat**](RangeFloat.md) |  | [optional]
**gsdCmPx** | [**RangeFloat**](RangeFloat.md) | [**RangeFloat**](RangeFloat.md) |  | [optional]
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)
