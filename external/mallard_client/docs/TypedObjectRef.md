# TypedObjectRef

Represents a reference to an object in the store, with an associated type.  Attributes:     id: The object ID.     type: The type of the object.

## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**id** | [**ObjectRef**](ObjectRef.md) |  |
**type** | [**ObjectType**](ObjectType.md) |  |

## Example

```python
from mallard_client.models.typed_object_ref import TypedObjectRef

# TODO update the JSON string below
json = "{}"
# create an instance of TypedObjectRef from a JSON string
typed_object_ref_instance = TypedObjectRef.from_json(json)
# print the JSON string representation of the object
print TypedObjectRef.to_json()

# convert the object into a dict
typed_object_ref_dict = typed_object_ref_instance.to_dict()
# create an instance of TypedObjectRef from a dict
typed_object_ref_form_dict = typed_object_ref.from_dict(typed_object_ref_dict)
```
[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)
