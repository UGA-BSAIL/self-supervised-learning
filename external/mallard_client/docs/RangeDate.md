# RangeDate

Specifies a range for numeric parameters in a query.  Attributes:     min_value: The minimum allowed value.     max_value: The maximum allowed value.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**min_value** | **date** |  | [optional]
**max_value** | **date** |  | [optional]

## Example

```python
from mallard_client.models.range_date import RangeDate

# TODO update the JSON string below
json = "{}"
# create an instance of RangeDate from a JSON string
range_date_instance = RangeDate.from_json(json)
# print the JSON string representation of the object
print RangeDate.to_json()

# convert the object into a dict
range_date_dict = range_date_instance.to_dict()
# create an instance of RangeDate from a dict
range_date_form_dict = range_date.from_dict(range_date_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
