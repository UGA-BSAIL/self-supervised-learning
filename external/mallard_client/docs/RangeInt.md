# RangeInt

Specifies a range for numeric parameters in a query.  Attributes:     min_value: The minimum allowed value.     max_value: The maximum allowed value.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**min_value** | **int** |  | [optional]
**max_value** | **int** |  | [optional]

## Example

```python
from mallard_client.models.range_int import RangeInt

# TODO update the JSON string below
json = "{}"
# create an instance of RangeInt from a JSON string
range_int_instance = RangeInt.from_json(json)
# print the JSON string representation of the object
print RangeInt.to_json()

# convert the object into a dict
range_int_dict = range_int_instance.to_dict()
# create an instance of RangeInt from a dict
range_int_form_dict = range_int.from_dict(range_int_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
