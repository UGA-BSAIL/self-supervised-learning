# mallard_client.model.typed_object_ref.TypedObjectRef

Represents a reference to an object in the store, with an associated type.  Attributes:     id: The object ID.     type: The type of the object.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Represents a reference to an object in the store, with an associated type.  Attributes:     id: The object ID.     type: The type of the object. |

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**id** | [**ObjectRef**](ObjectRef.md) | [**ObjectRef**](ObjectRef.md) |  |
**type** | [**ObjectType**](ObjectType.md) | [**ObjectType**](ObjectType.md) |  |
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)
