# RangeFloat

Specifies a range for numeric parameters in a query.  Attributes:     min_value: The minimum allowed value.     max_value: The maximum allowed value.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**min_value** | **float** |  | [optional]
**max_value** | **float** |  | [optional]

## Example

```python
from mallard_client.models.range_float import RangeFloat

# TODO update the JSON string below
json = "{}"
# create an instance of RangeFloat from a JSON string
range_float_instance = RangeFloat.from_json(json)
# print the JSON string representation of the object
print RangeFloat.to_json()

# convert the object into a dict
range_float_dict = range_float_instance.to_dict()
# create an instance of RangeFloat from a dict
range_float_form_dict = range_float.from_dict(range_float_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
