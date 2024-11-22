# mallard_client.model.ordering.Ordering

Represents an ordering that can be used for image data.  Attributes:     field: The field that we are ordering upon.     ascending: If true, sort ascending. Otherwise, sort descending.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Represents an ordering that can be used for image data.  Attributes:     field: The field that we are ordering upon.     ascending: If true, sort ascending. Otherwise, sort descending. |

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**field** | [**Field**](Field.md) | [**Field**](Field.md) |  |
**ascending** | bool,  | BoolClass,  |  | [optional] if omitted the server will use the default value of True
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)
