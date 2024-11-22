# mallard_client.model.uav_image_metadata.UavImageMetadata

Represents metadata for an image.  Attributes:     format: The format that the image is in. This will be deduced         automatically, but an expected format can be provided by the user         for verification.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Represents metadata for an image.  Attributes:     format: The format that the image is in. This will be deduced         automatically, but an expected format can be provided by the user         for verification. |

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**size** | decimal.Decimal, int,  | decimal.Decimal,  |  | [optional]
**name** | str,  | str,  |  | [optional]
**[platformType](#platformType)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] if omitted the server will use the default value of ground
**notes** | str,  | str,  |  | [optional] if omitted the server will use the default value of ""
**sessionName** | str,  | str,  |  | [optional]
**sequenceNumber** | decimal.Decimal, int,  | decimal.Decimal,  |  | [optional]
**captureDate** | str, date,  | str,  |  | [optional] value must conform to RFC-3339 full-date YYYY-MM-DD
**[location](#location)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] if omitted the server will use the default value of {}
**locationDescription** | str,  | str,  |  | [optional]
**camera** | str,  | str,  |  | [optional]
**altitudeMeters** | decimal.Decimal, int, float,  | decimal.Decimal,  |  | [optional]
**gsdCmPx** | decimal.Decimal, int, float,  | decimal.Decimal,  |  | [optional]
**format** | [**ImageFormat**](ImageFormat.md) | [**ImageFormat**](ImageFormat.md) |  | [optional]
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# platformType

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | if omitted the server will use the default value of ground

### Composed Schemas (allOf/anyOf/oneOf/not)
#### allOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[PlatformType](PlatformType.md) | [**PlatformType**](PlatformType.md) | [**PlatformType**](PlatformType.md) |  |

# location

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | if omitted the server will use the default value of {}

### Composed Schemas (allOf/anyOf/oneOf/not)
#### allOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[GeoPoint](GeoPoint.md) | [**GeoPoint**](GeoPoint.md) | [**GeoPoint**](GeoPoint.md) |  |

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)
