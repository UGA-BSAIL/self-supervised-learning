# Ordering

Represents an ordering that can be used for image data.  Attributes:     field: The field that we are ordering upon.     ascending: If true, sort ascending. Otherwise, sort descending.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**field** | [**Field**](Field.md) |  |
**ascending** | **bool** |  | [optional] [default to True]

## Example

```python
from mallard_client.models.ordering import Ordering

# TODO update the JSON string below
json = "{}"
# create an instance of Ordering from a JSON string
ordering_instance = Ordering.from_json(json)
# print the JSON string representation of the object
print Ordering.to_json()

# convert the object into a dict
ordering_dict = ordering_instance.to_dict()
# create an instance of Ordering from a dict
ordering_form_dict = ordering.from_dict(ordering_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
