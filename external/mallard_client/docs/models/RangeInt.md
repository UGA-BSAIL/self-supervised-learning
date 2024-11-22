# mallard_client.model.range_int.RangeInt

Specifies a range for numeric parameters in a query.  Attributes:     min_value: The minimum allowed value.     max_value: The maximum allowed value.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Specifies a range for numeric parameters in a query.  Attributes:     min_value: The minimum allowed value.     max_value: The maximum allowed value. |

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**minValue** | decimal.Decimal, int,  | decimal.Decimal,  |  | [optional]
**maxValue** | decimal.Decimal, int,  | decimal.Decimal,  |  | [optional]
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)
